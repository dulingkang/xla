from enum import Enum, auto
from typing import Union
from jax._src.lib import xla_extension as xe
# from jax.lib import xla_client as xc
from jax.lib import xla_bridge as xb
from jax.interpreters import mlir
import numpy as np
import xla_extension as xle
from xla.python import xla_client as xc

from type_conversion import conv_compileoptions, inv_conv_hlomodule


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


class HloStatus(Enum):
    """
    The status of an HloModule.
    See also the docstring at the beginning of shard_parallel/auto_sharding.py.
    """
    UNOPTIMIZED = auto()
    SHARDING_ANNOTATED = auto()
    SPMD_PARTITIONED = auto()
    FULLY_OPTIMIZED = auto()


class WrappedHlo:
    """Wrapped HloModule with HloStatus."""

    def __init__(self,
                 module: Union[xe.HloModule, xe.XlaComputation, bytes],
                 status: HloStatus = HloStatus.UNOPTIMIZED):
        if isinstance(module, xe.HloModule):
            self.module = module
        elif isinstance(module, xe.XlaComputation):
            self.module = module.get_hlo_module()
        else:
            assert isinstance(module, bytes)
            self.module = xe.XlaComputation(module).get_hlo_module()
        self.name = self.module.name
        self.status = status
        self.is_manually_annotated = False

    def get_computation(self) -> xe.XlaComputation:
        return xe.XlaComputation(self.module.as_serialized_hlo_module_proto())

    def get_mhlo(self):
        xla_computation = self.get_computation()
        module_str = xe.mlir.xla_computation_to_mlir_module(xla_computation)
        with mlir.make_ir_context():
            mhlo = mlir.ir.Module.parse(module_str)
        return mhlo

    def get_module(self) -> xe.HloModule:
        return self.module

    def get_hlo_proto(self):
        return self.module.as_serialized_hlo_module_proto()

    def program_shape(self):
        return self.module.program_shape()

    def set_input_shardings(self, sharding_protos):
        assert self.is_sharding_annotated() or self.is_unoptimized()
        xe.set_hlo_module_input_shardings(self.module, sharding_protos)

    def set_output_shardings(self, sharding_protos):
        assert self.is_sharding_annotated() or self.is_unoptimized()
        xe.set_hlo_module_output_shardings(self.module, sharding_protos)

    def is_unoptimized(self):
        return self.status == HloStatus.UNOPTIMIZED

    def is_sharding_annotated(self):
        return self.status == HloStatus.SHARDING_ANNOTATED

    def is_spmd_partitioned(self):
        return self.status == HloStatus.SPMD_PARTITIONED

    def to_string(self):
        return self.module.to_string()

    def __getstate__(self):
        return (self.get_hlo_proto(), self.status)

    def __setstate__(self, bytes_and_status):
        b, s = bytes_and_status
        self.__init__(b, s)


# hlomodule
hlo_text = """
ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default}
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
	'auto_sharding::simplify_graph': True,
	'auto_sharding::print_strategy': False,
	'auto_sharding::force_strategy': False,
	'auto_sharding::force_strategy_inst_indices': [0],
	'auto_sharding::force_strategy_stra_names': ["mesha"],
	'auto_sharding::rewrite_for_grad_acc': False,
	'auto_sharding::rewrite_indices': None,
	'combiner::all_gather_threshold': 1152921504606846976,
	'combiner::all_reduce_threshold': 1152921504606846976,  
}

compile_options2 = xle.CompileOptions.ParseFromString(b'\x1a\xe1\x01\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x1a\xba\x01\xf8\x01\x03\x98\x02\x01\xe0\x03\x01\xea\x032/usr/local/lib/python3.9/dist-packages/jaxlib/cuda\xb0\x04\x01\xb8\x04\x01\xc0\x04\x01\xb0\x06\x01\xc0\x07\x01\xc8\x07\x01\xd0\x07\x01\xd8\x07\x04\xf0\x07\x01\x88\x08\x01\xa0\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\xf0\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\xf8\x08\x01\xc0\t\x01\xe0\t\x01\xe8\t\x80\x80\x80\x0f\x80\n\x01\x98\n\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\xa8\n\x01\xb8\n\x80 \xc0\n\x01\xc8\n\x01\xd0\n\x01\xe0\n\x01\xe8\n\x01\xa8\x0b\x01\xc0\x0b\x01\xc8\x0b\x01\xe0\x0b\x01\xe8\x0b\x01\xf8\x0b\x01 \x01(\x020\x01J\x0e\x08\x01\x10\x02\x1a\x03\n\x01\x00\x1a\x03\n\x01\x01b\x01\x00')

compile_options3 = xle.CompileOptions.ParseFromString(b'\x1a\xe1\x01\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x1a\xba\x01\xf8\x01\x03\x98\x02\x01\xe0\x03\x01\xea\x032/usr/local/lib/python3.9/dist-packages/jaxlib/cuda\xb0\x04\x01\xb8\x04\x01\xc0\x04\x01\xb0\x06\x01\xc0\x07\x01\xc8\x07\x01\xd0\x07\x01\xd8\x07\x04\xf0\x07\x01\x88\x08\x01\xa0\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\xf0\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\xf8\x08\x01\xc0\t\x01\xe0\t\x01\xe8\t\x80\x80\x80\x0f\x80\n\x01\x98\n\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\xa8\n\x01\xb8\n\x80 \xc0\n\x01\xc8\n\x01\xd0\n\x01\xe0\n\x01\xe8\n\x01\xa8\x0b\x01\xc0\x0b\x01\xc8\x0b\x01\xe0\x0b\x01\xe8\x0b\x01\xf8\x0b\x01 \x01(\x020\x01J\x0e\x08\x01\x10\x02\x1a\x03\n\x01\x00\x1a\x03\n\x01\x01b\x01\x01')
# print(compile_options.argument_layouts)
# print(compile_options.parameter_is_tupled_arguments)  # False
# print(compile_options.executable_build_options)  # ExecutableBuildOptions{device_ordinal=-1, result_layout=nullopt, num_replicas=1
# print(compile_options.compile_portable_executable)  # False
# print(compile_options.profile_version)  # 0

with XlaPassContext(auto_sharding_options):
  # xle.run_auto_sharding(hlo_module, compile_options)
  # xle.run_spmd_partitioner(hlo_module, compile_options)
  backend = xc.make_cpu_client()
  mlir_module = xle.XlaComputation(hlo_module.as_serialized_hlo_module_proto()).as_hlo_text()
  backend.compile(mlir_module, compile_options)
