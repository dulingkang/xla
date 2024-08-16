/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>
#include <ostream>
#include <string>

#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_parser.h"
#include "xla/status.h"
#include "xla/tools/hlo_module_loader.h"
#include "tsl/platform/init_main.h"

// add by mesha
// #include "pybind11/pybind11.h"
#include "xla/hlo/experimental/auto_sharding/slice_auto_sharded_stages.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/compiler.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/dump.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/gather_expander.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/all_reduce_reassociate.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/spmd/redundant_slice_eliminator.h"
#include "xla/service/spmd/grad_acc_rewrite.h"
#include "xla/service/pass_context.h"
#include "xla/client/xla_computation.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/transforms/hlo_constant_splitter.h"
#include "xla/pjrt/pjrt_executable.h"


namespace xla {

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

namespace spmd {
// namespace {

const char kBeforeAutoShardingDumpName[] = "before_run_auto_sharding";
const char kBeforeSpmdPartitionDumpName[] = "before_run_spmd_partitioner";

Status RunAutoShardingPassFromFile(const std::string& file_name) {
  std::string hlo_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), file_name, &hlo_text));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      LoadModuleFromData(/*data=*/hlo_text, /*format=*/"hlo"));

  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSIGN_OR_RETURN(bool changed, AutoSharding(option).Run(hlo_module.get()));
  CHECK(changed);
  // std::cout << hlo_module->ToString() << std::endl;
  return OkStatus();
}

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

      // hhq
      // spmd_pipeline.AddPass<AutoSharding>();
      AutoShardingOption as_option;
      as_option.enable = pass_context::GetBool("auto_sharding::enable", true);
      as_option.memory_budget_per_device = pass_context::GetInt("auto_sharding::memory_budget_per_device", -1);
      as_option.force_all_gather_cost = pass_context::GetBool("auto_sharding::force_all_gather_cost", false);
      as_option.all_gather_cost = pass_context::GetDouble("auto_sharding::all_gather_cost");
      as_option.force_all_to_all_cost = pass_context::GetBool("auto_sharding::force_all_to_all_cost", false);
      as_option.all_to_all_cost = pass_context::GetDouble("auto_sharding::all_to_all_cost");
      as_option.allow_replicated_parameters = pass_context::GetBool("auto_sharding::allow_replicated_parameters", true);
      as_option.prefer_reduce_scatter = pass_context::GetBool("auto_sharding::prefer_reduce_scatter", false);
      as_option.reduce_scatter_grad_acc_friendly = pass_context::GetBool("auto_sharding::reduce_scatter_grad_acc_friendly", false);
      as_option.reduce_scatter_aggressive_partition = pass_context::GetBool("auto_sharding::reduce_scatter_aggressive_partition", false);
      as_option.batch_matmul_always_split_batch = pass_context::GetBool("auto_sharding::batch_matmul_always_split_batch", false);
      as_option.allow_recompute_heavy_op = pass_context::GetBool("auto_sharding::allow_recompute_heavy_op", true);
      as_option.allow_mixed_mesh_shape = pass_context::GetBool("auto_sharding::allow_mixed_mesh_shape", false);
      as_option.grad_acc_num_micro_batches = pass_context::GetInt("auto_sharding::grad_acc_num_micro_batches", 1);
      as_option.force_batch_dim_to_mesh_dim = pass_context::GetInt("auto_sharding::force_batch_dim_to_mesh_dim", -1);
      as_option.force_simple_heuristic = pass_context::GetString("auto_sharding::force_simple_heuristic", "");
      as_option.device_mesh_ids = pass_context::GetIntVector("auto_sharding::device_mesh_ids");
      as_option.device_mesh_shape = pass_context::GetIntVector("auto_sharding::device_mesh_shape");
      as_option.device_mesh_alpha = pass_context::GetDoubleVector("auto_sharding::device_mesh_alpha");
      as_option.device_mesh_beta = pass_context::GetDoubleVector("auto_sharding::device_mesh_beta");
      as_option.simplify_graph = pass_context::GetBool("auto_sharding::simplify_graph", true);
      as_option.force_strategy = pass_context::GetBool("auto_sharding::force_strategy", false);
      as_option.force_strategy_inst_indices = pass_context::GetIntVector("auto_sharding::force_strategy_inst_indices");
      as_option.force_strategy_stra_names = pass_context::GetStringVector("auto_sharding::force_strategy_stra_names");
      spmd_pipeline.AddPass<AutoSharding>(as_option);

      // hhq
      // spmd_pipeline.AddPass<ShardingPropagation>(
      //     /*is_spmd=*/true, /*propagate_metadata=*/false,
      //     /*allow_spmd_sharding_propagation_to_output=*/{true});
      spmd_pipeline.AddPass<ShardingPropagation>(/*is_spmd=*/true, /*propagate_metadata=*/false,
        /*allow_spmd_sharding_propagation_to_output=*/absl::Span<const bool>{true});

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
  // std::cout << hlo_module->ToString() << std::endl;
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
      spmd_pipeline.AddPass<GradAccRewrite>();  // should be used together with XLA_SKIP_NCCL_COLLECTIVE_IDS
    } else {
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }
  // std::cout << hlo_module->ToString() << std::endl;

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

// }  // namespace
}  // namespace spmd
}  // namespace xla

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

  tsl::port::InitMain("Run AutoSharding Pass", &argc, &argv);
  QCHECK(argc == 2) << "Must specify a single input file";
  TF_CHECK_OK(xla::spmd::RunAutoShardingPassFromFile(argv[1]));

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
  // for (xla::HloSharding x:hlo_module->spmd_parameters_shardings()) {
  //   std::cout << "spmd_parameters_shardings:" << x.ToString() << std::endl;
  // }

  return 0;
}
