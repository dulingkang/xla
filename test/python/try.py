from jax.lib import xla_extension as xe
from jax.lib import xla_client as xc
from jax.lib import xla_bridge as xb
import numpy as np


if __name__ == "__main__":
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
  import pdb; pdb.set_trace()
  print(type(compile_options))
  print(compile_options.argument_layouts)
  print(compile_options.parameter_is_tupled_arguments)  # False
  print(compile_options.executable_build_options)  # ExecutableBuildOptions{device_ordinal=-1, result_layout=nullopt, num_replicas=1
  print(compile_options.compile_portable_executable)  # False
  print(compile_options.profile_version)  # 0
  