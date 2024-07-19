# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")
load(
    "//tensorflow:tensorflow.bzl",
    "VERSION",
    "VERSION_MAJOR",
    "if_google",
    "if_oss",
    "tf_cc_shared_object",
    "tf_custom_op_library_additional_deps_impl",
    "tf_native_cc_binary",
)
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//tensorflow:tensorflow.default.bzl",
    "ADDITIONAL_API_INDEXABLE_SETTINGS",
    "tf_cc_shared_library",
)

# copybara:uncomment_begin
# load("//tools/build_defs/license:license.bzl", "license")
# copybara:uncomment_end

PACKAGE_STATIC_DEPS = [
    "//:__subpackages__",
    "@cudnn_frontend_archive//:__subpackages__",
    "@gif//:__subpackages__",
    "@highwayhash//:__subpackages__",
    "@hwloc//:__subpackages__",
    "@icu//:__subpackages__",
    "@libjpeg_turbo//:__subpackages__",
    "@llvm-project//:__subpackages__",
    "@nsync//:__subpackages__",
    "@png//:__subpackages__",
    "@sobol_data//:__subpackages__",
    "@stablehlo//:__subpackages__",
    "@tf_runtime//:__subpackages__",
    "@bazel_tools//:__subpackages__",
    "@boringssl//:__subpackages__",
    "@com_github_cares_cares//:__subpackages__",
    "@com_github_googlecloudplatform_tensorflow_gcp_tools//:__subpackages__",
    "@com_github_grpc_grpc//:__subpackages__",
    "@com_google_absl//:__subpackages__",
    "@com_google_googleapis//:__subpackages__",
    "@com_google_protobuf//:__subpackages__",
    "@com_googlesource_code_re2//:__subpackages__",
    "@compute_library//:__subpackages__",
    "@curl//:__subpackages__",
    "@double_conversion//:__subpackages__",
    "@eigen_archive//:__subpackages__",
    "@farmhash_archive//:__subpackages__",
    "@farmhash_gpu_archive//:__subpackages__",
    "@fft2d//:__subpackages__",
    "@gemmlowp//:__subpackages__",
    "@jsoncpp_git//:__subpackages__",
    "@libxsmm_archive//:__subpackages__",
    "@llvm_openmp//:__subpackages__",
    "@llvm_terminfo//:__subpackages__",
    "@llvm_zlib//:__subpackages__",
    "@lmdb//:__subpackages__",
    "@local_config_cuda//:__subpackages__",
    "@local_config_git//:__subpackages__",
    "@local_config_nccl//:__subpackages__",
    "@local_config_rocm//:__subpackages__",
    "@local_config_tensorrt//:__subpackages__",
    "@local_execution_config_platform//:__subpackages__",
    "@mkl_dnn_acl_compatible//:__subpackages__",
    "@mkl_dnn_v1//:__subpackages__",
    "@nccl_archive//:__subpackages__",
    "@org_sqlite//:__subpackages__",
    "@platforms//:__subpackages__",
    "@snappy//:__subpackages__",
    "@upb//:__subpackages__",
    "@zlib//:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = [":license"],
    default_visibility = [":internal"],
)

# copybara:uncomment_begin
# license(
#     name = "license",
#     package_name = "tensorflow",
# )
# copybara:uncomment_end

licenses(["notice"])

exports_files([
    "LICENSE",
    # The leakr files are used by //third_party/cloud_tpu and
    # //third_party/tensorboard/google:copybara_config_test.
    "leakr_badwords.dic",
    "leakr_badfiles.dic",
    "leakr_file_type_recipe.ftrcp",
])

# copybara:comment_begin(oss-only)
TENSORFLOW_API_INIT_FILES_V2 = (
    TENSORFLOW_API_INIT_FILES +
    get_compat_files(TENSORFLOW_API_INIT_FILES, 2) +
    get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
)

TENSORFLOW_API_INIT_FILES_V1 = (
    TENSORFLOW_API_INIT_FILES_V1 +
    get_compat_files(TENSORFLOW_API_INIT_FILES, 2) +
    get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
)
# copybara:comment_end

# Setting to use when loading kernels dynamically
config_setting(
    name = "dynamic_loaded_kernels",
    define_values = {
        "dynamic_loaded_kernels": "true",
        "framework_shared_object": "true",
    },
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon): Remove in favor of TSL version
config_setting(
    name = "windows",
    # Internal builds query the target OS.
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "windows"},
        {},
    ),
    # OSS builds query the CPU type.
    values = if_oss(
        {"cpu": "x64_windows"},
        {},
    ),
    visibility = ["//visibility:public"],
)