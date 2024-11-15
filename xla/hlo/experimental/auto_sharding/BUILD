# Automatic sharding annotation

load("//xla:xla.bzl", "xla_cc_binary", "xla_cc_test")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    packages = [
        "//platforms/...",
    ],
)

cc_library(
    name = "auto_sharding",
    srcs = [
        "auto_sharding.cc",
        "auto_sharding_dot_handler.cc",
        "slice_auto_sharded_stages.cc", # added by mesha
    ],
    hdrs = [
        "auto_sharding.h",
        "slice_auto_sharded_stages.h", # added by mesha
    ],
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_solver_option",
        ":auto_sharding_strategy",
        ":auto_sharding_util",
        ":cluster_environment",
        ":matrix",
        ":metrics",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:dump",
        "//xla/service:heap_simulator",
        "//xla/service:hlo_memory_scheduler",
        "//xla/service:hlo_ordering",
        "//xla/service:hlo_pass",
        "//xla/service:sharding_propagation",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_ortools//ortools/linear_solver",
        "@com_google_ortools//ortools/linear_solver:linear_solver_cc_proto",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
        "@pybind11", # added by mesha
    ],
)

cc_library(
    name = "auto_sharding_strategy",
    hdrs = [
        "auto_sharding_strategy.h",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "auto_sharding_cost_graph",
    hdrs = [
        "auto_sharding_cost_graph.h",
    ],
    deps = [
        ":auto_sharding_strategy",
        ":matrix",
    ],
)

cc_library(
    name = "matrix",
    hdrs = [
        "matrix.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "cluster_environment",
    srcs = ["cluster_environment.cc"],
    hdrs = ["cluster_environment.h"],
    deps = [
        ":auto_sharding_solver_option",
        ":auto_sharding_util",
        ":profiling_result",
        "//xla/hlo/ir:hlo",
        "//xla/service/spmd:spmd_partitioner",        
    ],
)

cc_library(
    name = "profiling_result",
    hdrs = ["profiling_result.h"],
    deps = [":auto_sharding_strategy"],
)

cc_library(
    name = "auto_sharding_solver_option",
    hdrs = ["auto_sharding_solver_option.h"],
)

cc_library(
    name = "auto_sharding_util",
    srcs = [
        "auto_sharding_util.cc",
    ],
    hdrs = [
        "auto_sharding_util.h",
    ],
    deps = [
        ":auto_sharding_strategy",
        "//xla:array",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:hlo_creation_utils", # added by mesha
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = ["@tsl//tsl/lib/monitoring:counter"],
)

cc_library(  # added by mesha: xla_cc_binary
    name = "auto_sharding_runner",
    srcs = ["auto_sharding_runner.cc"],
    ########## added by mesha ##########
    hdrs = ["auto_sharding_runner.h"],
    linkopts = [
        "-L/usr/local/lib/python3.9",
        "-lpython3.9",
        "-Wl,-rpath,/usr/local/lib/python3.9",
    ],
    visibility = ["//xla/python:__pkg__"],
    copts = ["-fexceptions"],
    ########## end added by mesha ##########       
    deps = [
        ":auto_sharding",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/stream_executor:dnn",
        "//xla/tools:hlo_module_loader",
        "@tsl//tsl/platform:platform_port",
        ########## added by mesha ##########
        "//xla/service:compiler",
        "//xla/service:sharding_remover",
        "//xla/service:algebraic_simplifier",
        "//xla/service:call_inliner",
        "//xla/service/gpu:gpu_conv_rewriter",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service:hlo_pass",
        "//xla/service:sort_simplifier",
        "//xla/service:dump",
        "//xla/service:dot_decomposer",
        "//xla/service:zero_sized_hlo_elimination",
        "//xla/service:conditional_canonicalizer",
        "//xla/service:tuple_simplifier",
        "//xla/service:scatter_expander",
        "//xla/service:gather_expander",
        "//xla/service:hlo_cse",
        "//xla/service:sharding_propagation",
        "//xla/service:hlo_verifier",
        "//xla/service:cpu_gpu_shape_verifier",  
        "//xla/service:while_loop_constant_sinking",  
        "//xla/service:while_loop_simplifier",  
        "//xla/service:reshape_mover",  
        "//xla/service:hlo_constant_folding",  
        "//xla/service:conditional_simplifier",  
        "//xla/service:transpose_folding",  
        "//xla/service:all_reduce_reassociate",  
        "//xla/service/gpu:matmul_utils",  
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
        "//xla/service/spmd:grad_acc_rewrite",  
        "//xla/service/spmd:redundant_slice_eliminator",   
        "//xla/service:pass_context",
        "//xla/client:executable_build_options",
        "//xla/hlo/transforms:hlo_constant_splitter",
        "//xla/pjrt:pjrt_executable",
        "//xla/client:xla_computation",       
        ########## end added by mesha ##########        
    ],
)

########## added by mesha ##########
xla_cc_binary(
  name = "auto_sharding_runner_test",
  srcs = ["auto_sharding_runner.cc"],
  # compatible_with = get_compatible_with_libtpu_portable(),
  deps = [
      ":auto_sharding_runner",
  ],
)
########## end added by mesha ##########

build_test(
    name = "auto_sharding_runner_build_test",
    targets = [
        ":auto_sharding_runner",
    ],
)

xla_cc_test(
    name = "auto_sharding_test",
    srcs = ["auto_sharding_test.cc"],
    ########## added by mesha ##########
    linkopts = [
        "-L/usr/local/lib/python3.9",
        "-lpython3.9",
        "-Wl,-rpath,/usr/local/lib/python3.9",
    ],
    ########## end added by mesha ##########       
    deps = [
        ":auto_sharding",
        ":auto_sharding_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_ortools//ortools/linear_solver",
        "@tsl//tsl/lib/core:status_test_util",
        "@llvm-project//llvm:Support",  # added by mesha
    ],
)
