from jax._src.lib import xla_client as xc, xla_extension as xe
from jax.lib import xla_bridge as xb
import numpy as np
import xla_extension as xle


def conv_hlomodule(hlo_module: xe.HloModule) -> xle.HloModule:
  """
  Convert xe HloModule to xle HloModule.
  """
  b = hlo_module.as_serialized_hlo_module_proto()
  return xle.HloModule.from_serialized_hlo_module_proto(b)


def inv_conv_hlomodule(hlo_module: xle.HloModule) -> xe.HloModule:
  """
  Convert xle HloModule to xe HloModule.
  """
  b = hlo_module.as_serialized_hlo_module_proto()
  return xe.HloModule.from_serialized_hlo_module_proto(b)


def conv_compileoptions(compile_options: xe.CompileOptions) -> xle.CompileOptions:
  """
  Convert xe CompileOptions to xle CompileOptions.
  """
  s = compile_options.SerializeAsString()
  return xle.CompileOptions.ParseFromString(s)


def inv_conv_compileoptions(compile_options: xle.CompileOptions) -> xe.CompileOptions:
  """
  Convert xle CompileOptions to xe CompileOptions.
  """
  s = compile_options.SerializeAsString()
  return xe.CompileOptions.ParseFromString(s)


# def conv_client(client: xe.Client) -> xle.Client:
#   """
#   Convert xe client to xle client.
#   """
#   s = client.serialize_executable()
#   return xle.Client.deserialize_executable(s)

          
# def inv_conv_client(client: xle.Client) -> xe.Client:
#   """
#   Convert xle client to xe client.
#   """
#   s = client.serialize_executable()
#   return xe.Client.deserialize_executable(s)


if __name__ == "__main__":
  # hlomodule
  hlo_text = """
    HloModule module
    ENTRY %elementwise {
      %param0 = f32[16,32,64]{2,1,0} parameter(0)
      %param1 = f32[16,32,64]{2,1,0} parameter(1)
      ROOT root = f32[16,32,64]{2,1,0} add(%param0, %param1)
    }
    """
  hlo_module = xe.hlo_module_from_text(hlo_text)
  print("hlo_module", type(hlo_module))
  hlo_module_new = conv_hlomodule(hlo_module)
  print("hlo_module_new", type(hlo_module_new))
  hlo_module = inv_conv_hlomodule(hlo_module_new)
  print("hlo_module", type(hlo_module))

  # compile options
  num_replicas = 1
  num_partitions = 1
  device_assignment = np.arange(1).reshape((1, -1))
  use_spmd_partitioning = True

  compile_options = xb.get_compile_options(
      num_replicas=num_replicas,
      num_partitions=num_partitions,
      device_assignment=device_assignment,
      use_spmd_partitioning=use_spmd_partitioning,
  )
  print("compile_options", type(compile_options))    
  compile_options_new = conv_compileoptions(compile_options)
  print("compile_options_new", type(compile_options_new))    
  compile_options = inv_conv_compileoptions(compile_options_new)
  print("compile_options", type(compile_options))
