#include "xla/service/spmd/alpa_compiler.h"
#include "xla/service/spmd/auto_sharding.h"
#include "xla/service/spmd/auto_sharding_util.h"
#include "xla/service/spmd/grad_acc_rewrite.h"
#include "xla/service/spmd/redundant_slice_eliminator.h"
#include "xla/service/spmd/slice_auto_sharded_stages.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"

// Copied from gpu_compiler.cc
#include "xla/hlo/transforms/hlo_constant_splitter.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/all_reduce_reassociate.h"
#include "xla/service/call_inliner.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/dot_merger.h"
#include "xla/service/dump.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/tools/hlo_module_loader.h"


namespace xla {

namespace {
// Adds the HloVerifier for GPU to the given pipeline.
void AddHloVerifier(HloPassPipeline* pipeline, HloVerifierOpts&& opts = {},
                    bool debug_only = false) {
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}

bool ConvIsLowerable(HloInstruction* conv) {
  return gpu::GpuConvRewriter::ConvIsLowerable(conv);
}
}  // namespace

namespace spmd {

const char kBeforeAutoShardingDumpName[] = "before_run_auto_sharding";
const char kBeforeSpmdPartitionDumpName[] = "before_run_spmd_partitioner";

// TODO(yonghao): Check correctness of compile options and modules
Status PreCompileCheck(const CompileOptions& options) {
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  if (build_options.has_device_assignment()) {
    if (build_options.device_assignment().replica_count() !=
        build_options.num_replicas()) {
      return InvalidArgument(
          "Mismatched number of replicas for device "
          "assignment and computation (%d vs %d).\n%s",
          build_options.device_assignment().replica_count(),
          build_options.num_replicas(),
          build_options.device_assignment().ToString());
    }
    // TODO(yonghao): for TPU, computation count != 1 is unsupported
    if (build_options.device_assignment().computation_count() !=
        build_options.num_partitions()) {
      return InvalidArgument(
          "Mismatched number of partitions for device "
          "assignment and computation (%d vs %d).\n%s",
          build_options.device_assignment().computation_count(),
          build_options.num_partitions(),
          build_options.device_assignment().ToString());
    }
  }

  return OkStatus();
}

StatusOr<HloModuleConfig> CreateHloModuleConfig(const HloModule* hlo_module,
                                                const CompileOptions options) {
  TF_RETURN_IF_ERROR(PreCompileCheck(options));

  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const ProgramShape& program_shape =
      hlo_module->entry_computation_layout().ComputeProgramShape();
  const ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  // hhq
  // TF_ASSIGN_OR_RETURN(
  //     auto module_config,
  //     HloModule::CreateModuleConfigFromShape(
  //         program_shape, build_options.debug_options(), &execution_options));
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  TF_ASSIGN_OR_RETURN(
      auto module_config,
      HloModule::CreateModuleConfigFromShape(
          program_shape, debug_options, &execution_options));

  return module_config;
}

Status RunAutoShardingPass(HloModule* hlo_module,
                           const CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      CreateHloModuleConfig(hlo_module, options));
  hlo_module->set_config(module_config);
  DumpHloModuleIfEnabled(*hlo_module, kBeforeAutoShardingDumpName);

  // TODO(yonghao): TF Profiler Traceme
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts({},
                                                             ConvIsLowerable);
  // "slow" minmax means we propagate nan.
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      !debug_options.xla_gpu_enable_fast_min_max());
  layout_insensitive_algsimp_opts.set_enable_dot_strength_reduction(false);  // Added by Alpa

  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("run-auto-sharding");
    AddHloVerifier(&spmd_pipeline);
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      // Run some IR cleanup passes before running the SPMD partitioning
      // passes.
      spmd_pipeline.AddPass<CallInliner>();
      spmd_pipeline.AddPass<DotDecomposer>();  // Added by Alpa
      spmd_pipeline.AddPass<ZeroSizedHloElimination>();
      spmd_pipeline.AddPass<ConditionalCanonicalizer>();

      HloPassPipeline& spmd_simplify =
          spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

      spmd_simplify.AddPass<AlgebraicSimplifier>(
          layout_insensitive_algsimp_opts);

      spmd_simplify.AddPass<SortSimplifier>();
      spmd_simplify.AddPass<TupleSimplifier>();
      // spmd_simplify.AddPass<ScatterSimplifier>();
      spmd_simplify.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);
      // spmd_simplify.AddPass<GatherSimplifier>();
      spmd_simplify.AddPass<GatherExpander>(
          GatherExpander::kEliminateSimpleGathers);
      spmd_simplify.AddPass<WhileLoopConstantSinking>();
      spmd_simplify.AddPass<WhileLoopSimplifier>();

      spmd_simplify.AddPass<ReshapeMover>();
      spmd_simplify.AddPass<HloConstantFolding>();
      spmd_simplify.AddPass<ConditionalSimplifier>();
      spmd_simplify.AddPass<TransposeFolding>(
          gpu::CanFoldTransposeOperandIntoDot);  // Added by Alpa
      spmd_simplify.AddPass<HloCSE>(
          /*is_layout_sensitive=*/false);  // Added by Alpa
      spmd_simplify.AddPass<HloDCE>();
      spmd_pipeline.AddPass<HloConstantSplitter>();
      spmd_pipeline.AddPass<AutoSharding>();

      // hhq
      // spmd_pipeline.AddPass<ShardingPropagation>(
      //     /*is_spmd=*/true, /*propagate_metadata=*/false,
      //     /*allow_spmd_sharding_propagation_to_output=*/true);
      spmd_pipeline.AddPass<ShardingPropagation>(/*is_spmd=*/true, /*propagate_metadata=*/false,
                                                 /*allow_spmd_sharding_propagation_to_output=*/absl::Span<const bool>{true});

      std::cout << "hhq5" << std::endl;

      spmd_pipeline.AddPass<SliceAutoShardedStages>();
    } else {
      spmd_pipeline.AddPass<CallInliner>();
      spmd_pipeline.AddPass<SliceAutoShardedStages>();
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }
  std::cout << hlo_module->ToString() << std::endl;
  return OkStatus();
}

Status RunSpmdPartitionerPass(HloModule* hlo_module,
                              const CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      CreateHloModuleConfig(hlo_module, options));
  hlo_module->set_config(module_config);

  DumpHloModuleIfEnabled(*hlo_module, kBeforeSpmdPartitionDumpName);

  // TODO(yonghao): TF Profiler Traceme
  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("run-spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      // hhq
      // spmd_pipeline.AddPass<ShardingPropagation>(
      //     /*is_spmd=*/true, /*propagate_metadata=*/false,
      //     /*allow_spmd_sharding_propagation_to_output=*/true);
      spmd_pipeline.AddPass<ShardingPropagation>(/*is_spmd=*/true, /*propagate_metadata=*/false,
                                                 /*allow_spmd_sharding_propagation_to_output=*/absl::Span<const bool>{true});
      spmd_pipeline.AddPass<StatefulRngSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
      spmd_pipeline.AddPass<RedundantSliceEliminator>();
      spmd_pipeline.AddPass<AllReduceReassociate>();
      spmd_pipeline.AddPass<GradAccRewrite>();
    } else {
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }
  std::cout << hlo_module->ToString() << std::endl;
  return OkStatus();
}

Status SetHloModuleOutputShardings(
    HloModule* module, const std::vector<OpSharding>& op_shardings) {
  // Run some simplification passes to remove redundant tuples.
  // Otherwise, these redundant tuples and other custom call markers together
  // will make the propagation generate unexpected results.
  HloPassPipeline pipeline("set-sharding-pipeline");
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());

  // Set the sharding for the output tuple
  HloComputation* entry = module->entry_computation();
  HloInstruction* output_tuple = entry->root_instruction();

  ShapeTree<HloSharding> tuple_sharding(output_tuple->shape(),
                                        HloSharding::Replicate());
  CHECK_EQ(tuple_sharding.leaf_count(), op_shardings.size());

  size_t i = 0;
  for (auto& leaf : tuple_sharding.leaves()) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(op_shardings[i++]));
    leaf.second = hlo_sharding;
  }
  output_tuple->set_sharding(HloSharding::Tuple(tuple_sharding));

  if (IsPassThroughTuple(output_tuple)) {
    // Also set the operand. Otherwise, the propagation generates unexpected
    // results.
    output_tuple->mutable_operand(0)->set_sharding(
        HloSharding::Tuple(tuple_sharding));
  }

  return OkStatus();
}

Status SetHloModuleInputShardings(HloModule* module,
                                  const std::vector<OpSharding>& op_shardings) {
  // Run some simplification passes to remove redundant tuples.
  // Otherwise, these redundant tuples and other custom call markers together
  // will make the propagation generate unexpected results.
  HloPassPipeline pipeline("set-sharding-pipeline");
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());

  HloComputation* entry = module->entry_computation();
  // hhq
  // std::vector<HloInstruction*> input_insts = entry->parameter_instructions();
  HloInstruction::InstructionVector input_insts = entry->parameter_instructions();

  CHECK_EQ(input_insts.size(), op_shardings.size());

  size_t i = 0;
  for (auto& inst : input_insts) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(op_shardings[i++]));
    if (IsUndefined(hlo_sharding)) {
      continue;
    }
    inst->set_sharding(HloSharding::Single(inst->shape(), hlo_sharding));
  }

  return OkStatus();
}

};  // namespace spmd
};  // namespace xla


int main(int argc, char** argv) {
//   const std::string& hlo_text = R"(I0521 12:04:45.883483    1509 service.cc:186] HloModule test_log_stripping
// I0521 12:04:45.883483    1509 service.cc:186]
// I0521 12:04:45.883483    1509 service.cc:186] ENTRY entry {
// I0521 12:04:45.883483    1509 service.cc:186]   p0 = f32[4]{0} parameter(0)
// I0521 12:04:45.883483    1509 service.cc:186]   p1 = f32[4]{0} parameter(1)
// I0521 12:04:45.883483    1509 service.cc:186]   add = f32[4]{0} add(p0, p1)
// I0521 12:04:45.883483    1509 service.cc:186]   ROOT rooty = (f32[4]{0}, f32[4]{0}) tuple(p1, add)
// I0521 12:04:45.883483    1509 service.cc:186] })";  

//   const std::string& hlo_text = R"(
// HloModule module
// ENTRY matmul {
//   parameter.1 = f32[32,64]{1,0} parameter(0)
//   parameter.2 = f32[64,128]{1,0} parameter(1)
//   ROOT root = f32[32,128]{1,0} dot(parameter.1, parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// })";

  const std::string& hlo_text = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[16,32,64]{2,1,0} parameter(0)
  %param1 = f32[16,32,64]{2,1,0} parameter(1)
  ROOT root = f32[16,32,64]{2,1,0} add(%param0, %param1)
})";

  std::cout << "Test RunAutoShardingPass...\n" << std::endl;
  absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_ptr = xla::LoadModuleFromData(/*data=*/hlo_text, /*format=*/"hlo");
  xla::HloModule* hlo_module;
  if (hlo_module_ptr.ok()) {
    hlo_module = hlo_module_ptr->get();
  } else {
    std::cout << "Failed to load HloModule: " << hlo_module_ptr.status() << std::endl;
    return -1;
  }

  xla::ExecutableBuildOptions build_options = xla::ExecutableBuildOptions();
  build_options.set_device_ordinal(0);
  build_options.set_num_replicas(1);
  build_options.set_num_partitions(2);
  build_options.set_use_spmd_partitioning(true);

  xla::CompileOptions options = {};
  options.compile_portable_executable = false;
  options.parameter_is_tupled_arguments = false;
  options.profile_version = 0;
  options.executable_build_options = build_options;

  TF_CHECK_OK(xla::spmd::RunAutoShardingPass(hlo_module, options));

  std::cout << "Test RunSpmdPartitionerPass...\n" << std::endl;
  TF_CHECK_OK(xla::spmd::RunSpmdPartitionerPass(hlo_module, options));

  std::cout << "spmd_output_sharding:" << hlo_module->spmd_output_sharding().ToString() << std::endl;

  return 0;
}
