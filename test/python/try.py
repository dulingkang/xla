# from jax.lib import xla_extension as xe
# from jax.lib import xla_client as xc
from jax.lib import xla_bridge as xb
import numpy as np
import xla_extension as xle

from type_conversion import conv_compileoptions


class XlaPassContext:
    """A global context for passing arguments from python to XLA c++ passes."""

    current = None

    def __init__(self, value_dict):
        self.value_dict = value_dict

    def __enter__(self):
        assert XlaPassContext.current is None, ("Do not support nested context")
        XlaPassContext.current = self
        xle.set_pass_context(self.value_dict)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        XlaPassContext.current = None
        xle.clear_pass_context()


auto_sharding_options = {
	'auto_sharding::enable': True,
	'auto_sharding::memory_budget_per_device': -1,
	'auto_sharding::force_all_gather_cost': False,
	'auto_sharding::all_gather_cost': 10000000000000.0,
	'auto_sharding::force_all_to_all_cost': False,
	'auto_sharding::all_to_all_cost': 10000000000000.0,
	'auto_sharding::allow_replicated_parameters': True,
	'auto_sharding::prefer_reduce_scatter': False,
	'auto_sharding::reduce_scatter_grad_acc_friendly': False,
	'auto_sharding::reduce_scatter_aggressive_partition': False,
	'auto_sharding::batch_matmul_always_split_batch': True,
	'auto_sharding::allow_recompute_heavy_op': False,
	'auto_sharding::allow_mixed_mesh_shape': False,
	'auto_sharding::grad_acc_num_micro_batches': 1,
	'auto_sharding::force_batch_dim_to_mesh_dim': -1,
	'auto_sharding::force_simple_heuristic': '',
	'auto_sharding::device_mesh_ids': (0, 1),
	'auto_sharding::device_mesh_shape': (1, 2),
	'auto_sharding::device_mesh_alpha': (1.0, 1.0),
	'auto_sharding::device_mesh_beta': (1.0, 0.1),
	'auto_sharding::device_mesh_prof_result': None,
	'auto_sharding::rewrite_for_grad_acc': False,
	'auto_sharding::rewrite_indices': None,
	'combiner::all_gather_threshold': 1152921504606846976,
	'combiner::all_reduce_threshold': 1152921504606846976,
	'auto_sharding::simplify_graph': True,
	'auto_sharding::print_strategy': False,
	'auto_sharding::force_strategy': False,
 
	'auto_sharding::force_strategy_inst_indices': [0],
	'auto_sharding::force_strategy_stra_names': ["mesha"]
}


# hlomodule
hlo_text = """
  HloModule module
  ENTRY %elementwise {
    %param0 = f32[16,32,64]{2,1,0} parameter(0)
    %param1 = f32[16,32,64]{2,1,0} parameter(1)
    ROOT root = f32[16,32,64]{2,1,0} add(%param0, %param1)
  }
  """
hlo_module = xle.hlo_module_from_text(hlo_text)

# compile_options
num_replicas = 1
num_partitions = 2
device_assignment = np.arange(num_partitions).reshape((1, -1))
use_spmd_partitioning = True

compile_options = xb.get_compile_options(
  num_replicas=num_replicas,
  num_partitions=num_partitions,
  device_assignment=device_assignment,
  use_spmd_partitioning=use_spmd_partitioning,
)
compile_options = conv_compileoptions(compile_options)
# print(type(compile_options))
# print(compile_options.argument_layouts)
# print(compile_options.parameter_is_tupled_arguments)  # False
# print(compile_options.executable_build_options)  # ExecutableBuildOptions{device_ordinal=-1, result_layout=nullopt, num_replicas=1
# print(compile_options.compile_portable_executable)  # False
# print(compile_options.profile_version)  # 0

with XlaPassContext(auto_sharding_options):
  xle.run_auto_sharding(hlo_module, compile_options)
  xle.run_spmd_partitioner(hlo_module, compile_options)
