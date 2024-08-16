"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    # ABSL_COMMIT = "b971ac5250ea8de900eae9f95e06548d14cd95fe"
    # ABSL_COMMIT = "273292d1cfc0a94a65082ee350509af1d113344d"
    # ABSL_SHA256 = "94aef187f688665dc299d09286bfa0d22c4ecb86a80b156dff6aabadc5a5c26d"
    ABSL_COMMIT = "fb3621f4f897824c0dbe0615fa94543df6192f30"
    ABSL_SHA256 = "0320586856674d16b0b7a4d4afb22151bdc798490bb7f295eddd8f6a62b46fea"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/abseil-cpp.cmake)

    SYS_DIRS = [
        "algorithm",
        "base",
        "cleanup",
        "container",
        "debugging",
        "flags",
        "functional",
        "hash",
        "memory",
        "meta",
        "numeric",
        "random",
        "status",
        "strings",
        "synchronization",
        "time",
        "types",
        "utility",
    ]
    SYS_LINKS = {
        "//third_party/absl:system.absl.{name}.BUILD".format(name = n): "absl/{name}/BUILD.bazel".format(name = n)
        for n in SYS_DIRS
    }

    tf_http_archive(
        name = "com_google_absl",
        sha256 = ABSL_SHA256,
        build_file = "//third_party/absl:com_google_absl.BUILD",
        system_build_file = "//third_party/absl:system.BUILD",
        system_link_files = SYS_LINKS,
        # This patch pulls in a fix for designated initializers that MSVC
        # complains about. It shouldn't be necessary at the next LTS release.
        patch_file = ["//third_party/absl:absl_designated_initializers.patch"],
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT)),
    )
