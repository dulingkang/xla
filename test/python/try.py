from enum import Enum, auto
from typing import Union
from jax._src.lib import xla_extension as xe
# from jax.lib import xla_client as xc
from jax.lib import xla_bridge as xb
from jax.interpreters import mlir
import numpy as np
import xla_extension as xle

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
HloModule train_step_shard_parallel, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias), {3}: (3, {}, may-alias), {4}: (4, {}, may-alias), {5}: (5, {}, may-alias), {6}: (6, {}, may-alias), {7}: (7, {}, may-alias), {8}: (8, {}, may-alias) }, entry_computation_layout={(s32[],f32[32]{0},f32[3,3,3,32]{3,2,1,0},f32[10]{0},f32[32768,10]{1,0},f32[32]{0},f32[3,3,3,32]{3,2,1,0},f32[10]{0},f32[32768,10]{1,0},f32[100,32,32,3]{3,2,1,0},s32[100]{0})->(s32[], f32[32]{0}, f32[3,3,3,32]{3,2,1,0}, f32[10]{0}, f32[32768,10]{1,0}, /*index=5*/f32[32]{0}, f32[3,3,3,32]{3,2,1,0}, f32[10]{0}, f32[32768,10]{0,1}, f32[])}

%region_0.62 (Arg_0.63: f32[], Arg_1.64: f32[]) -> f32[] {
  %Arg_0.63 = f32[] parameter(0)
  %Arg_1.64 = f32[] parameter(1)
  ROOT %maximum.65 = f32[] maximum(f32[] %Arg_0.63, f32[] %Arg_1.64), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_max[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
}

%region_1.73 (Arg_0.74: f32[], Arg_1.75: f32[]) -> f32[] {
  %Arg_0.74 = f32[] parameter(0)
  %Arg_1.75 = f32[] parameter(1)
  ROOT %add.76 = f32[] add(f32[] %Arg_0.74, f32[] %Arg_1.75), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
}

%region_2.85 (Arg_0.86: f32[], Arg_1.87: f32[]) -> f32[] {
  %Arg_0.86 = f32[] parameter(0)
  %Arg_1.87 = f32[] parameter(1)
  ROOT %add.88 = f32[] add(f32[] %Arg_0.86, f32[] %Arg_1.87), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
}

%region_3.91 (Arg_0.92: f32[], Arg_1.93: f32[]) -> f32[] {
  %Arg_0.92 = f32[] parameter(0)
  %Arg_1.93 = f32[] parameter(1)
  ROOT %add.94 = f32[] add(f32[] %Arg_0.92, f32[] %Arg_1.93), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
}

%region_4.100 (Arg_0.101: f32[], Arg_1.102: f32[]) -> f32[] {
  %Arg_0.101 = f32[] parameter(0)
  %Arg_1.102 = f32[] parameter(1)
  ROOT %add.103 = f32[] add(f32[] %Arg_0.101, f32[] %Arg_1.102), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
}

%region_5.107 (Arg_0.108: f32[], Arg_1.109: f32[]) -> f32[] {
  %Arg_0.108 = f32[] parameter(0)
  %Arg_1.109 = f32[] parameter(1)
  ROOT %add.110 = f32[] add(f32[] %Arg_0.108, f32[] %Arg_1.109), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
}

%region_6.115 (Arg_0.116: f32[], Arg_1.117: f32[]) -> f32[] {
  %Arg_0.116 = f32[] parameter(0)
  %Arg_1.117 = f32[] parameter(1)
  ROOT %add.118 = f32[] add(f32[] %Arg_0.116, f32[] %Arg_1.117), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=131}
}

%region_7.124 (Arg_0.125: f32[], Arg_1.126: f32[]) -> f32[] {
  %Arg_0.125 = f32[] parameter(0)
  %Arg_1.126 = f32[] parameter(1)
  ROOT %add.127 = f32[] add(f32[] %Arg_0.125, f32[] %Arg_1.126), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=234}
}

ENTRY %main.148 (Arg_0.1: s32[], Arg_1.2: f32[32], Arg_2.3: f32[3,3,3,32], Arg_3.4: f32[10], Arg_4.5: f32[32768,10], Arg_5.6: f32[32], Arg_6.7: f32[3,3,3,32], Arg_7.8: f32[10], Arg_8.9: f32[32768,10], Arg_9.10: f32[100,32,32,3], Arg_10.11: s32[100]) -> (s32[], f32[32], f32[3,3,3,32], f32[10], f32[32768,10], /*index=5*/f32[32], f32[3,3,3,32], f32[10], f32[32768,10], f32[]) {
  %Arg_0.1 = s32[] parameter(0)
  %constant.33 = s32[] constant(1)
  %add.146 = s32[] add(s32[] %Arg_0.1, s32[] %constant.33), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/mesha/mesha/training/train_state.py" source_line=75}
  %Arg_1.2 = f32[32]{0} parameter(1)
  %Arg_10.11 = s32[100]{0} parameter(10)
  %reshape.50 = s32[100,1]{1,0} reshape(s32[100]{0} %Arg_10.11), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/broadcast_in_dim[shape=(100, 1) broadcast_dimensions=(0,)]" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %broadcast.53 = s32[100,1]{1,0} broadcast(s32[100,1]{1,0} %reshape.50), dimensions={0,1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/eq" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %reshape.54 = s32[100]{0} reshape(s32[100,1]{1,0} %broadcast.53), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/eq" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %broadcast.55 = s32[100,10]{1,0} broadcast(s32[100]{0} %reshape.54), dimensions={0}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/eq" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %iota.51 = s32[10]{0} iota(), iota_dimension=0, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/iota[dtype=int32 shape=(1, 10) dimension=1]" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %reshape.52 = s32[1,10]{1,0} reshape(s32[10]{0} %iota.51), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/iota[dtype=int32 shape=(1, 10) dimension=1]" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %broadcast.56 = s32[1,10]{1,0} broadcast(s32[1,10]{1,0} %reshape.52), dimensions={0,1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/eq" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %reshape.57 = s32[10]{0} reshape(s32[1,10]{1,0} %broadcast.56), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/eq" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %broadcast.58 = s32[100,10]{1,0} broadcast(s32[10]{0} %reshape.57), dimensions={1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/eq" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %compare.59 = pred[100,10]{1,0} compare(s32[100,10]{1,0} %broadcast.55, s32[100,10]{1,0} %broadcast.58), direction=EQ, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/eq" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %convert.60 = f32[100,10]{1,0} convert(pred[100,10]{1,0} %compare.59), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=43}
  %constant.28 = f32[] constant(-0.01)
  %broadcast.29 = f32[100,10]{1,0} broadcast(f32[] %constant.28), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/broadcast_in_dim[shape=(100, 10) broadcast_dimensions=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %multiply.98 = f32[100,10]{1,0} multiply(f32[100,10]{1,0} %convert.60, f32[100,10]{1,0} %broadcast.29), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %negate.99 = f32[100,10]{1,0} negate(f32[100,10]{1,0} %multiply.98), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/neg" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %constant.35 = f32[] constant(0)
  %reduce.104 = f32[100]{0} reduce(f32[100,10]{1,0} %negate.99, f32[] %constant.35), dimensions={1}, to_apply=%region_4.100, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reshape.105 = f32[100,1]{1,0} reshape(f32[100]{0} %reduce.104), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reshape[new_sizes=(100, 1) dimensions=None]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %Arg_9.10 = f32[100,32,32,3]{3,2,1,0} parameter(9)
  %Arg_2.3 = f32[3,3,3,32]{3,2,1,0} parameter(2)
  %convolution.37 = f32[100,32,32,32]{3,2,1,0} convolution(f32[100,32,32,3]{3,2,1,0} %Arg_9.10, f32[3,3,3,32]{3,2,1,0} %Arg_2.3), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/conv_general_dilated[window_strides=(1, 1) padding=((1, 1), (1, 1)) lhs_dilation=(1, 1) rhs_dilation=(1, 1) dimension_numbers=ConvDimensionNumbers(lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), out_spec=(0, 3, 1, 2)) feature_group_count=1 batch_group_count=1 precision=None preferred_element_type=None]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=222}
  %reshape.38 = f32[1,1,1,32]{3,2,1,0} reshape(f32[32]{0} %Arg_1.2), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reshape[new_sizes=(1, 1, 1, 32) dimensions=None]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=234}
  %broadcast.39 = f32[1,1,1,32]{3,2,1,0} broadcast(f32[1,1,1,32]{3,2,1,0} %reshape.38), dimensions={0,1,2,3}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=234}
  %reshape.40 = f32[32]{0} reshape(f32[1,1,1,32]{3,2,1,0} %broadcast.39), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=234}
  %broadcast.41 = f32[100,32,32,32]{3,2,1,0} broadcast(f32[32]{0} %reshape.40), dimensions={3}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=234}
  %add.42 = f32[100,32,32,32]{3,2,1,0} add(f32[100,32,32,32]{3,2,1,0} %convolution.37, f32[100,32,32,32]{3,2,1,0} %broadcast.41), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=234}
  %reshape.43 = f32[100,32768]{1,0} reshape(f32[100,32,32,32]{3,2,1,0} %add.42), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reshape[new_sizes=(100, 32768) dimensions=None]" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=33}
  %Arg_4.5 = f32[32768,10]{1,0} parameter(4)
  %dot.44 = f32[100,10]{1,0} dot(f32[100,32768]{1,0} %reshape.43, f32[32768,10]{1,0} %Arg_4.5), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/dot_general[dimension_numbers=(((1,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=127}
  %Arg_3.4 = f32[10]{0} parameter(3)
  %reshape.45 = f32[1,10]{1,0} reshape(f32[10]{0} %Arg_3.4), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reshape[new_sizes=(1, 10) dimensions=None]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=131}
  %broadcast.46 = f32[1,10]{1,0} broadcast(f32[1,10]{1,0} %reshape.45), dimensions={0,1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=131}
  %reshape.47 = f32[10]{0} reshape(f32[1,10]{1,0} %broadcast.46), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=131}
  %broadcast.48 = f32[100,10]{1,0} broadcast(f32[10]{0} %reshape.47), dimensions={1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=131}
  %add.49 = f32[100,10]{1,0} add(f32[100,10]{1,0} %dot.44, f32[100,10]{1,0} %broadcast.48), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=131}
  %constant.31 = f32[] constant(1e-07)
  %broadcast.32 = f32[100,10]{1,0} broadcast(f32[] %constant.31), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %add.61 = f32[100,10]{1,0} add(f32[100,10]{1,0} %add.49, f32[100,10]{1,0} %broadcast.32), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %constant.36 = f32[] constant(-inf)
  %reduce.66 = f32[100]{0} reduce(f32[100,10]{1,0} %add.61, f32[] %constant.36), dimensions={1}, to_apply=%region_0.62, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_max[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reshape.67 = f32[100,1]{1,0} reshape(f32[100]{0} %reduce.66), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/broadcast_in_dim[shape=(100, 1) broadcast_dimensions=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %broadcast.68 = f32[100,1]{1,0} broadcast(f32[100,1]{1,0} %reshape.67), dimensions={0,1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reshape.69 = f32[100]{0} reshape(f32[100,1]{1,0} %broadcast.68), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %broadcast.70 = f32[100,10]{1,0} broadcast(f32[100]{0} %reshape.69), dimensions={0}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %subtract.71 = f32[100,10]{1,0} subtract(f32[100,10]{1,0} %add.61, f32[100,10]{1,0} %broadcast.70), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %exponential.72 = f32[100,10]{1,0} exponential(f32[100,10]{1,0} %subtract.71), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/exp" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reduce.77 = f32[100]{0} reduce(f32[100,10]{1,0} %exponential.72, f32[] %constant.35), dimensions={1}, to_apply=%region_1.73, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reshape.78 = f32[100,1]{1,0} reshape(f32[100]{0} %reduce.77), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/broadcast_in_dim[shape=(100, 1) broadcast_dimensions=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %divide.106 = f32[100,1]{1,0} divide(f32[100,1]{1,0} %reshape.105, f32[100,1]{1,0} %reshape.78), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/div" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reduce.111 = f32[100]{0} reduce(f32[100,1]{1,0} %divide.106, f32[] %constant.35), dimensions={1}, to_apply=%region_5.107, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %broadcast.112 = f32[100,10]{1,0} broadcast(f32[100]{0} %reduce.111), dimensions={0}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/broadcast_in_dim[shape=(100, 10) broadcast_dimensions=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %multiply.113 = f32[100,10]{1,0} multiply(f32[100,10]{1,0} %broadcast.112, f32[100,10]{1,0} %exponential.72), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %add.114 = f32[100,10]{1,0} add(f32[100,10]{1,0} %multiply.98, f32[100,10]{1,0} %multiply.113), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add_any" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %dot.122 = f32[100,32768]{1,0} dot(f32[100,10]{1,0} %add.114, f32[32768,10]{1,0} %Arg_4.5), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/dot_general[dimension_numbers=(((1,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=127}
  %reshape.123 = f32[100,32,32,32]{3,2,1,0} reshape(f32[100,32768]{1,0} %dot.122), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reshape[new_sizes=(100, 32, 32, 32) dimensions=None]" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=33}
  %reduce.128 = f32[32]{0} reduce(f32[100,32,32,32]{3,2,1,0} %reshape.123, f32[] %constant.35), dimensions={0,1,2}, to_apply=%region_7.124, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=234}
  %Arg_5.6 = f32[32]{0} parameter(5)
  %constant.26 = f32[] constant(0.9)
  %broadcast.27 = f32[32]{0} broadcast(f32[] %constant.26), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %multiply.130 = f32[32]{0} multiply(f32[32]{0} %Arg_5.6, f32[32]{0} %broadcast.27), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %add.131 = f32[32]{0} add(f32[32]{0} %reduce.128, f32[32]{0} %multiply.130), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %constant.18 = f32[] constant(-0.01)
  %broadcast.19 = f32[32]{0} broadcast(f32[] %constant.18), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=518}
  %multiply.138 = f32[32]{0} multiply(f32[32]{0} %add.131, f32[32]{0} %broadcast.19), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=518}
  %add.142 = f32[32]{0} add(f32[32]{0} %Arg_1.2, f32[32]{0} %multiply.138), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/update.py" source_line=43}
  %convolution.129 = f32[3,3,3,32]{3,2,1,0} convolution(f32[100,32,32,3]{3,2,1,0} %Arg_9.10, f32[100,32,32,32]{3,2,1,0} %reshape.123), window={size=32x32 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/conv_general_dilated[window_strides=(1, 1) padding=((1, 1), (1, 1)) lhs_dilation=(1, 1) rhs_dilation=(1, 1) dimension_numbers=ConvDimensionNumbers(lhs_spec=(3, 0, 1, 2), rhs_spec=(3, 0, 1, 2), out_spec=(2, 3, 0, 1)) feature_group_count=1 batch_group_count=1 precision=None preferred_element_type=None]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/conv.py" source_line=222}
  %Arg_6.7 = f32[3,3,3,32]{3,2,1,0} parameter(6)
  %constant.24 = f32[] constant(0.9)
  %broadcast.25 = f32[3,3,3,32]{3,2,1,0} broadcast(f32[] %constant.24), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %multiply.132 = f32[3,3,3,32]{3,2,1,0} multiply(f32[3,3,3,32]{3,2,1,0} %Arg_6.7, f32[3,3,3,32]{3,2,1,0} %broadcast.25), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %add.133 = f32[3,3,3,32]{3,2,1,0} add(f32[3,3,3,32]{3,2,1,0} %convolution.129, f32[3,3,3,32]{3,2,1,0} %multiply.132), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %constant.16 = f32[] constant(-0.01)
  %broadcast.17 = f32[3,3,3,32]{3,2,1,0} broadcast(f32[] %constant.16), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=518}
  %multiply.139 = f32[3,3,3,32]{3,2,1,0} multiply(f32[3,3,3,32]{3,2,1,0} %add.133, f32[3,3,3,32]{3,2,1,0} %broadcast.17), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=518}
  %add.143 = f32[3,3,3,32]{3,2,1,0} add(f32[3,3,3,32]{3,2,1,0} %Arg_2.3, f32[3,3,3,32]{3,2,1,0} %multiply.139), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/update.py" source_line=43}
  %reduce.119 = f32[10]{0} reduce(f32[100,10]{1,0} %add.114, f32[] %constant.35), dimensions={0}, to_apply=%region_6.115, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=131}
  %Arg_7.8 = f32[10]{0} parameter(7)
  %constant.22 = f32[] constant(0.9)
  %broadcast.23 = f32[10]{0} broadcast(f32[] %constant.22), dimensions={}
  %multiply.134 = f32[10]{0} multiply(f32[10]{0} %Arg_7.8, f32[10]{0} %broadcast.23), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %add.135 = f32[10]{0} add(f32[10]{0} %reduce.119, f32[10]{0} %multiply.134), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %constant.14 = f32[] constant(-0.01)
  %broadcast.15 = f32[10]{0} broadcast(f32[] %constant.14), dimensions={}
  %multiply.140 = f32[10]{0} multiply(f32[10]{0} %add.135, f32[10]{0} %broadcast.15), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=518}
  %add.144 = f32[10]{0} add(f32[10]{0} %Arg_3.4, f32[10]{0} %multiply.140), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/update.py" source_line=43}
  %dot.120 = f32[10,32768]{1,0} dot(f32[100,10]{1,0} %add.114, f32[100,32768]{1,0} %reshape.43), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/dot_general[dimension_numbers=(((0,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=127}
  %transpose.121 = f32[32768,10]{0,1} transpose(f32[10,32768]{1,0} %dot.120), dimensions={1,0}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/transpose[permutation=(1, 0)]" source_file="/data/hejing/distri/neurai/neurai/nn/layer/linear.py" source_line=127}
  %Arg_8.9 = f32[32768,10]{1,0} parameter(8)
  %constant.20 = f32[] constant(0.9)
  %broadcast.21 = f32[32768,10]{1,0} broadcast(f32[] %constant.20), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %multiply.136 = f32[32768,10]{1,0} multiply(f32[32768,10]{1,0} %Arg_8.9, f32[32768,10]{1,0} %broadcast.21), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %add.137 = f32[32768,10]{0,1} add(f32[32768,10]{0,1} %transpose.121, f32[32768,10]{1,0} %multiply.136), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=70}
  %constant.12 = f32[] constant(-0.01)
  %broadcast.13 = f32[32768,10]{1,0} broadcast(f32[] %constant.12), dimensions={}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=518}
  %multiply.141 = f32[32768,10]{0,1} multiply(f32[32768,10]{0,1} %add.137, f32[32768,10]{1,0} %broadcast.13), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/transform.py" source_line=518}
  %add.145 = f32[32768,10]{1,0} add(f32[32768,10]{1,0} %Arg_4.5, f32[32768,10]{0,1} %multiply.141), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/add" source_file="/usr/local/lib/python3.9/dist-packages/optax/_src/update.py" source_line=43}
  %log.79 = f32[100,1]{1,0} log(f32[100,1]{1,0} %reshape.78), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/log" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %broadcast.80 = f32[100,1]{1,0} broadcast(f32[100,1]{1,0} %log.79), dimensions={0,1}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reshape.81 = f32[100]{0} reshape(f32[100,1]{1,0} %broadcast.80), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %broadcast.82 = f32[100,10]{1,0} broadcast(f32[100]{0} %reshape.81), dimensions={0}, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %subtract.83 = f32[100,10]{1,0} subtract(f32[100,10]{1,0} %subtract.71, f32[100,10]{1,0} %broadcast.82), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/sub" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %multiply.84 = f32[100,10]{1,0} multiply(f32[100,10]{1,0} %convert.60, f32[100,10]{1,0} %subtract.83), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/mul" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reduce.89 = f32[100]{0} reduce(f32[100,10]{1,0} %multiply.84, f32[] %constant.35), dimensions={1}, to_apply=%region_2.85, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(1,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %negate.90 = f32[100]{0} negate(f32[100]{0} %reduce.89), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/neg" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %reduce.95 = f32[] reduce(f32[100]{0} %negate.90, f32[] %constant.35), dimensions={0}, to_apply=%region_3.91, metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/reduce_sum[axes=(0,)]" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %constant.34 = f32[] constant(100)
  %divide.96 = f32[] divide(f32[] %reduce.95, f32[] %constant.34), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/div" source_file="/data/hejing/distri/neurai/neurai/nn/loss.py" source_line=48}
  %constant.30 = f32[] constant(1)
  %divide.97 = f32[] divide(f32[] %divide.96, f32[] %constant.30), metadata={op_name="parallelize(train_step_shard_parallel)/jit(main)/div" source_file="/data/hejing/distri/mesha/tests/shard_parallel/test_cnn.py" source_line=44}
  ROOT %tuple.147 = (s32[], f32[32]{0}, f32[3,3,3,32]{3,2,1,0}, f32[10]{0}, f32[32768,10]{1,0}, /*index=5*/f32[32]{0}, f32[3,3,3,32]{3,2,1,0}, f32[10]{0}, f32[32768,10]{0,1}, f32[]) tuple(s32[] %add.146, f32[32]{0} %add.142, f32[3,3,3,32]{3,2,1,0} %add.143, f32[10]{0} %add.144, f32[32768,10]{1,0} %add.145, /*index=5*/f32[32]{0} %add.131, f32[3,3,3,32]{3,2,1,0} %add.133, f32[10]{0} %add.135, f32[32768,10]{0,1} %add.137, f32[] %divide.97)
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
  xle.run_auto_sharding(hlo_module, compile_options)
  # xle.run_spmd_partitioner(hlo_module, compile_options2)
  # xle.Client.compile(xle.XlaComputation(hlo_module.as_serialized_hlo_module_proto()).as_hlo_text(), compile_options3)

