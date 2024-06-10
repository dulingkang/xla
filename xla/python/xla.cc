/* Copyright 2019 The OpenXLA Authors.

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

#include <Python.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"  // from @nanobind
#include "nanobind/nb_defs.h"  // from @nanobind
#include "nanobind/stl/function.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/optional.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/set.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // from @nanobind  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // from @nanobind  // IWYU pragma: keep
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt_proxy/client/py_module.h"
#include "xla/python/py_client.h"
#include "xla/python/py_program.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/tsl/python/lib/core/numpy.h"  //NOLINT
#ifdef XLA_PYTHON_ENABLE_GPU
#include "xla/python/gpu_support.h"
#endif  // XLA_PYTHON_ENABLE_GPU

#ifdef __linux__
#include "gloo/transport/tcp/attr.h"  // from @gloo
#include "gloo/transport/tcp/device.h"  // from @gloo
#include "xla/pjrt/cpu/gloo_collectives.h"
#include "xla/pjrt/cpu/gloo_kv_store.h"
#endif  // __linux__

#if !defined(_WIN32) && !defined(PLATFORM_GOOGLE)
#include "xla/pjrt/cpu/mpi_collectives.h"
#endif  // !_WIN32 && !PLATFORM_GOOGLE

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/custom_call_sharding.h"
#include "xla/python/dlpack.h"
#include "xla/python/jax_jit.h"
#include "xla/python/logging.h"  // IWYU pragma: keep
#include "xla/python/mlir.h"
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_class_ptr.h"
#include "xla/python/ops.h"
#include "xla/python/outfeed_receiver_py.h"
#include "xla/python/pjit.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pmap_lib.h"
#include "xla/python/pprof_profile_builder.h"
#include "xla/python/profiler.h"
#include "xla/python/py_array.h"
#include "xla/python/py_compile_only_client.h"
#include "xla/python/py_device.h"
#include "xla/python/py_device_list.h"
#include "xla/python/py_executable.h"
#include "xla/python/py_memory_space.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/pytree.h"
#include "xla/python/sharding.h"
#include "xla/python/traceback.h"
#include "xla/python/transfer_guard_lib.h"
#include "xla/python/weakref_lru_cache.h"
#include "xla/python/xla_compiler.h"
#include "xla/tsl/distributed_runtime/preemption/preemption_sync_manager.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/status.h"

// TODO(phawkins): remove host_id properties after JAX is update to avoid them.

namespace xla {
namespace {

namespace nb = nanobind;

bool IsOptimizedBuild() {
#if NDEBUG
  return true;
#else
  return false;
#endif  // NDEBUG
}

// Is*san reports whether the build is under that particular sanitizer.
bool IsAsan() {
#if defined(ADDRESS_SANITIZER)
  return true;
#else  // defined(ADDRESS_SANITIZER)
  return false;
#endif
}

bool IsMsan() {
#if defined(MEMORY_SANITIZER)
  return true;
#else  // defined(MEMORY_SANITIZER)
  return false;
#endif
}

bool IsTsan() {
#if defined(THREAD_SANITIZER)
  return true;
#else  // defined(THREAD_SANITIZER)
  return false;
#endif
}

// IsSanitized reports whether the build is under any sanitizer.
bool IsSanitized() { return IsAsan() || IsMsan() || IsTsan(); }

}  // namespace

NB_MODULE(xla_extension, m_nb) {
  // Initialize ABSL logging because code within XLA uses it.
#ifndef PLATFORM_GOOGLE
  InitializeAbslLogging();
#endif  // PLATFORM_GOOGLE

  // We seem to get a fair number of leak warnings from nanobind. It's unclear
  // whether these are false positives or not.
  nb::set_leak_warnings(false);

  tsl::ImportNumpy();

  // Exceptions
  nb::exception<XlaRuntimeError> xla_runtime_error(m_nb, "XlaRuntimeError",
                                                   PyExc_RuntimeError);

  // Types
  nb::enum_<PrimitiveType>(m_nb, "PrimitiveType")
      .value("PRIMITIVE_TYPE_INVALID", PRIMITIVE_TYPE_INVALID)
      .value("PRED", PRED)
      .value("S4", S4)
      .value("S8", S8)
      .value("S16", S16)
      .value("S32", S32)
      .value("S64", S64)
      .value("U4", U4)
      .value("U8", U8)
      .value("U16", U16)
      .value("U32", U32)
      .value("U64", U64)
      .value("F16", F16)
      .value("F8E4M3FN", F8E4M3FN)
      .value("F8E4M3B11FNUZ", F8E4M3B11FNUZ)
      .value("F8E4M3FNUZ", F8E4M3FNUZ)
      .value("F8E5M2", F8E5M2)
      .value("F8E5M2FNUZ", F8E5M2FNUZ)
      .value("BF16", BF16)
      .value("F32", F32)
      .value("F64", F64)
      .value("C64", C64)
      .value("C128", C128)
      .value("TUPLE", TUPLE)
      .value("OPAQUE_TYPE", OPAQUE_TYPE)
      .value("TOKEN", TOKEN);

  // Must be before PyClient.compile.
  BuildXlaCompilerSubmodule(m_nb);

  PyDevice::RegisterPythonType(m_nb);
  PyMemorySpace::RegisterPythonType(m_nb);
  PyClient::RegisterPythonTypes(m_nb);

  nb::class_<PjRtLayout>(m_nb, "PjRtLayout")
      .def("__str__", &PjRtLayout::ToString)
      .def("__eq__", [](const PjRtLayout& layout,
                        const PjRtLayout& other) { return layout == other; })
      .def("__hash__",
           [](const PjRtLayout& layout) { return absl::HashOf(layout); });

  nb::class_<PjRtXlaLayout, PjRtLayout>(m_nb, "PjRtXlaLayout")
      .def("__getstate__",
           [](const PjRtXlaLayout& layout) -> nb::tuple {
             absl::StatusOr<std::string> serialized = layout.Serialize();
             ThrowIfError(serialized.status());
             return nb::make_tuple(
                 nb::bytes(serialized->data(), serialized->size()));
           })
      .def("__setstate__", [](PjRtXlaLayout* self, nb::tuple t) {
        // TODO(b/328671718): don't assume PjRtXlaLayout. We probably want a
        // generic method on PjRtCompiler instead, although we'll have
        // somehow have to attach a compiler to this PjRtLayout (something
        // like ClientAndPtr).
        nb::bytes serialized = nb::cast<nb::bytes>(t[0]);
        absl::StatusOr<PjRtXlaLayout> layout = PjRtXlaLayout::Deserialize(
            std::string_view(serialized.c_str(), serialized.size()));
        ThrowIfError(layout.status());
        new (self) PjRtXlaLayout(std::move(*layout));
      });

  jax::BuildWeakrefLRUCacheAPI(m_nb);

  nb::class_<xla::cpu::CollectivesInterface> cpu_collectives(m_nb,
                                                             "CpuCollectives");

  m_nb.def(
      "make_gloo_tcp_collectives",
      [](std::shared_ptr<DistributedRuntimeClient> distributed_client,

         std::optional<std::string> hostname,
         std::optional<std::string> interface)
          -> std::shared_ptr<xla::cpu::CollectivesInterface> {
#ifdef __linux__
        std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;
        if (distributed_client != nullptr) {
          kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"cpu:");
        }
        auto gloo_kv_store = std::make_unique<cpu::GlooKeyValueStore>(kv_store);
        auto tcp_attrs = gloo::transport::tcp::attr();
        if (hostname) {
          tcp_attrs.hostname = *hostname;
        }
        if (interface) {
          tcp_attrs.iface = *interface;
        }
        auto tcp_device = gloo::transport::tcp::CreateDevice(tcp_attrs);
        return std::make_shared<cpu::GlooCollectives>(std::move(gloo_kv_store),
                                                      std::move(tcp_device));
#else   // __linux__
        throw xla::XlaRuntimeError(
            "make_gloo_tcp_collectives only implemented for linux");
#endif  // __linux__
      },
      nb::arg("distributed_client"), nb::arg("hostname").none() = std::nullopt,
      nb::arg("interface").none() = std::nullopt);

#if !defined(_WIN32) && !defined(PLATFORM_GOOGLE)
  nb::class_<cpu::MpiCollectives> mpi_collectives(m_nb, "MpiCollectives",
                                                  cpu_collectives);
  mpi_collectives.def("Init", &cpu::MpiCollectives::Init);
  mpi_collectives.def("Finalize", &cpu::MpiCollectives::Finalize);
  m_nb.def("make_mpi_collectives",
           []() -> std::shared_ptr<cpu::MpiCollectives> {
             return std::make_shared<cpu::MpiCollectives>();
           });
#else   // !_WIN32 && !PLATFORM_GOOGLE
  m_nb.def("make_mpi_collectives",
           []() -> std::shared_ptr<xla::cpu::CollectivesInterface> {
             throw xla::XlaRuntimeError(
                 "make_mpi_collectives is not implemented for Windows");
           });
#endif  // !_WIN32 && !PLATFORM_GOOGLE

  m_nb.def(
      "get_tfrt_cpu_client",
      [](bool asynchronous,
         std::shared_ptr<DistributedRuntimeClient> distributed_client,
         int node_id, int num_nodes,
         std::shared_ptr<xla::cpu::CollectivesInterface> collectives)
          -> nb_class_ptr<PyClient> {
        std::unique_ptr<ifrt::PjRtClient> ifrt_client;
        {
          nb::gil_scoped_release gil_release;
          CpuClientOptions options;

          options.asynchronous = asynchronous;
          options.collectives = std::move(collectives);
          options.process_id = node_id;
          std::unique_ptr<PjRtClient> client =
              xla::ValueOrThrow(GetTfrtCpuClient(options));
          ifrt::PjRtClient::CreateOptions ifrt_options;
          ifrt_options.pjrt_client =
              std::shared_ptr<PjRtClient>(std::move(client));
          if (distributed_client != nullptr) {
            ifrt_options.kv_store =
                GetDistributedKeyValueStore(distributed_client,
                                            /*key_prefix=*/"cpu:");
            ifrt_options.process_id = node_id;
            ifrt_options.num_processes = num_nodes;
          }
          ifrt_client =
              ValueOrThrow(ifrt::PjRtClient::Create(std::move(ifrt_options)));
        }
        return PyClient::Make(std::move(ifrt_client));
      },
      nb::arg("asynchronous") = true, nb::arg("distributed_client") = nullptr,
      nb::arg("node_id") = 0, nb::arg("num_nodes") = 1,
      nb::arg("collectives").none() =
          std::shared_ptr<xla::cpu::CollectivesInterface>());
  m_nb.def("pjrt_plugin_loaded", [](std::string platform_name) -> bool {
    absl::StatusOr<const PJRT_Api*> pjrt_api = pjrt::PjrtApi(platform_name);
    return pjrt_api.ok();
  });
  m_nb.def(
      "load_pjrt_plugin",
      [](std::string platform_name, std::optional<std::string> library_path,
         std::optional<nb::capsule> c_api) -> nb::capsule {
        if (library_path.has_value()) {
          const PJRT_Api* api = xla::ValueOrThrow(
              pjrt::LoadPjrtPlugin(platform_name, *library_path));
          return nb::capsule(absl::bit_cast<void*>(api), "pjrt_c_api");
        }
        if (absl::string_view(c_api->name()) != "pjrt_c_api") {
          throw nb::value_error(
              "c_api argument to load_pjrt_plugin is not a pjrt_c_api "
              "capsule.");
        }
        xla::ThrowIfError(pjrt::SetPjrtApi(
            platform_name, static_cast<const PJRT_Api*>(c_api->data())));
        return *c_api;
      },
      nb::arg("platform_name"), nb::arg("library_path").none() = std::nullopt,
      nb::arg("c_api").none() = std::nullopt);
  m_nb.def("pjrt_plugin_initialized", [](std::string platform_name) -> bool {
    return xla::ValueOrThrow(pjrt::IsPjrtPluginInitialized(platform_name));
  });
  m_nb.def("initialize_pjrt_plugin", [](std::string platform_name) {
    return xla::ThrowIfError(pjrt::InitializePjrtPlugin(platform_name));
  });

#ifdef XLA_PYTHON_ENABLE_GPU
  RegisterGpuClientAndDefineGpuAllocatorConfig(m_nb);
#endif  // XLA_PYTHON_ENABLE_GPU

  m_nb.def(
      "get_c_api_client",
      [](std::string platform_name,
         const absl::flat_hash_map<std::string, PjRtValueType>& options,
         std::shared_ptr<DistributedRuntimeClient> distributed_client)
          -> nb_class_ptr<PyClient> {
        std::unique_ptr<ifrt::PjRtClient> ifrt_client;
        {
          nb::gil_scoped_release gil_release;
          std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;
          if (distributed_client != nullptr) {
            kv_store = GetDistributedKeyValueStore(
                distributed_client,
                /*key_prefix=*/absl::StrCat(platform_name, ":"));
          }
          std::unique_ptr<PjRtClient> c_api_client = xla::ValueOrThrow(
              GetCApiClient(platform_name, options, kv_store));
          ifrt_client = ifrt::PjRtClient::Create(std::move(c_api_client));
        }
        return PyClient::Make(std::move(ifrt_client));
      },
      nb::arg("platform_name"),
      nb::arg("options") = absl::flat_hash_map<std::string, PjRtValueType>(),
      nb::arg("distributed_client").none() = nullptr);
  // TODO(b/322357665): Delete this method after TPU plugin changes to use the
  // standard registration.
  m_nb.def("get_default_c_api_topology",
           [](std::string platform_name, std::string topology_name,
              const absl::flat_hash_map<std::string, PjRtValueType>& options)
               -> std::shared_ptr<ifrt::Topology> {
             return std::make_shared<ifrt::PjRtTopology>(xla::ValueOrThrow(
                 GetCApiTopology(platform_name, topology_name, options)));
           });
  m_nb.def(
      "get_c_api_topology",
      [](nb::capsule c_api, std::string topology_name,
         const absl::flat_hash_map<std::string, PjRtValueType>& options)
          -> std::shared_ptr<ifrt::Topology> {
        if (absl::string_view(c_api.name()) != "pjrt_c_api") {
          throw nb::value_error(
              "Argument to get_c_api_topology was not a pjrt_c_api capsule.");
        }
        return std::make_shared<ifrt::PjRtTopology>(xla::ValueOrThrow(
            GetCApiTopology(static_cast<const PJRT_Api*>(c_api.data()),
                            topology_name, options)));
      });
  m_nb.def("get_topology_for_devices",
           [](const std::vector<nb_class_ptr<PyDevice>>& py_devices) {
             if (py_devices.empty()) {
               throw nb::value_error(
                   "get_topology_for_devices requires >= 1 devices.");
             }
             auto client = py_devices[0]->client();
             ifrt::DeviceList::Devices ifrt_devices;
             ifrt_devices.reserve(py_devices.size());
             for (const auto& py_device : py_devices) {
               if (py_device->client().get() != client.get()) {
                 throw nb::value_error(
                     "devices passed to get_topology_for_devices come from "
                     "different clients.");
               }
               ifrt_devices.push_back(py_device->device());
             }
             ifrt::DeviceList device_list(std::move(ifrt_devices));
             return xla::ValueOrThrow(
                 client->ifrt_client()->GetTopologyForDevices(device_list));
           });

  TF_CHECK_OK(PyArray::RegisterTypes(m_nb));
  jax::RegisterDeviceList(m_nb);
  jax::RegisterSharding(m_nb);

  nb::class_<CompiledMemoryStats>(m_nb, "CompiledMemoryStats")
      .def_rw("generated_code_size_in_bytes",
              &CompiledMemoryStats::generated_code_size_in_bytes)
      .def_rw("argument_size_in_bytes",
              &CompiledMemoryStats::argument_size_in_bytes)
      .def_rw("output_size_in_bytes",
              &CompiledMemoryStats::output_size_in_bytes)
      .def_rw("alias_size_in_bytes", &CompiledMemoryStats::alias_size_in_bytes)
      .def_rw("temp_size_in_bytes", &CompiledMemoryStats::temp_size_in_bytes)
      .def_rw("host_generated_code_size_in_bytes",
              &CompiledMemoryStats::host_generated_code_size_in_bytes)
      .def_rw("host_argument_size_in_bytes",
              &CompiledMemoryStats::host_argument_size_in_bytes)
      .def_rw("host_output_size_in_bytes",
              &CompiledMemoryStats::host_output_size_in_bytes)
      .def_rw("host_alias_size_in_bytes",
              &CompiledMemoryStats::host_alias_size_in_bytes)
      .def_rw("host_temp_size_in_bytes",
              &CompiledMemoryStats::host_temp_size_in_bytes)
      .def_prop_ro("serialized_hlo_proto",
                   [](const CompiledMemoryStats& cms) -> nb::bytes {
                     return nb::bytes(cms.serialized_hlo_proto.data(),
                                      cms.serialized_hlo_proto.size());
                   })
      .def("__str__", &CompiledMemoryStats::DebugString);

  nb::class_<PyExecuteResults>(m_nb, "ExecuteResults")
      .def("__len__", [](PyExecuteResults& results) { return results.Size(); })
      .def("disassemble_into_single_device_arrays",
           &PyExecuteResults::DisassembleIntoSingleDeviceArrays)
      .def("disassemble_prefix_into_single_device_arrays",
           &PyExecuteResults::DisassemblePrefixIntoSingleDeviceArrays)
      .def("consume_with_handlers", &PyExecuteResults::ConsumeWithHandlers)
      .def("consume_token", &PyExecuteResults::ConsumeToken);

  nb::class_<PyLoadedExecutable>(m_nb, "LoadedExecutable")
      .def_prop_ro("client", &PyLoadedExecutable::client)
      .def("local_logical_device_ids",
           [](PyLoadedExecutable* exec) {
             auto span = exec->addressable_device_logical_ids();
             // Not on dispatch critical path, so ok to have heap allocation.
             std::vector<std::pair<int, int>> addressable_device_logic_ids;
             addressable_device_logic_ids.reserve(span.size());
             for (const auto& logical_device_id : span) {
               addressable_device_logic_ids.push_back(std::make_pair(
                   logical_device_id.replica, logical_device_id.partition));
             }
           })
      .def("local_devices", &PyLoadedExecutable::AddressableDevices)
      .def("size_of_generated_code_in_bytes",
           &PyLoadedExecutable::SizeOfGeneratedCodeInBytes)
      .def(
          "get_compiled_memory_stats",
          xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetCompiledMemoryStats))
      .def("delete", &PyLoadedExecutable::Delete)
      .def("execute_sharded_on_local_devices",
           xla::ValueOrThrowWrapper(
               &PyLoadedExecutable::ExecuteShardedOnLocalDevices),
           nb::arg("arguments"))
      .def("execute_sharded_on_local_devices_with_tokens",
           xla::ValueOrThrowWrapper(
               &PyLoadedExecutable::ExecuteShardedOnLocalDevicesWithTokens),
           nb::arg("arguments"))
      // TODO(parkers): Switch execute_sharded_on_local_devices* to this.
      .def("execute_sharded",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::ExecuteSharded),
           nb::arg("arguments"), nb::arg("with_tokens") = false)
      .def("hlo_modules", ValueOrThrowWrapper(&PyLoadedExecutable::HloModules))
      .def("get_output_memory_kinds",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetOutputMemoryKinds))
      .def("get_output_shardings", &PyLoadedExecutable::GetOutputShardings)
      .def("get_parameter_layouts",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetParameterLayouts))
      .def("get_output_layouts",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetOutputLayouts))
      .def("get_parameter_shardings",
           &PyLoadedExecutable::GetParameterShardings)
      .def("keep_alive", &PyLoadedExecutable::KeepAlive)
      .def("compile_options",
           [](const PyLoadedExecutable& self) {
             return xla::ValueOrThrow(
                 self.pjrt_executable()->GetCompileOptions());
           })
      .def("cost_analysis",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetCostAnalysis))
      .def_prop_ro("traceback", &PyLoadedExecutable::traceback)
      .def_prop_ro("fingerprint", [](PyLoadedExecutable* exec) -> nb::object {
        if (exec->fingerprint().has_value()) {
          return nb::bytes(exec->fingerprint()->data(),
                           exec->fingerprint()->size());
        } else {
          return nb::none();
        }
      });
  nb::class_<PyToken> token(m_nb, "Token");
  token.def("block_until_ready",
            [](PyToken& self) { xla::ThrowIfError(self.Await()); });

  nb::class_<PyShardedToken> sharded_token(m_nb, "ShardedToken");
  sharded_token.def("block_until_ready", [](PyShardedToken& self) {
    xla::ThrowIfError(self.Await());
  });
  sharded_token.def("get_token", &PyShardedToken::GetPyToken);

  m_nb.def("buffer_to_dlpack_managed_tensor",
           xla::ValueOrThrowWrapper(BufferToDLPackManagedTensor),
           nb::arg("buffer"), nb::arg("stream").none() = nb::none());
  m_nb.def(
      "dlpack_managed_tensor_to_buffer",
      [](const nb::capsule& tensor, nb_class_ptr<PyDevice> device,
         std::optional<std::intptr_t> stream) {
        return xla::ValueOrThrow(DLPackManagedTensorToBuffer(
            tensor, device->device(), device->client(), stream));
      },
      nb::arg("dlpack"), nb::arg("device"), nb::arg("stream").none());
  // Legacy overload
  m_nb.def(
      "dlpack_managed_tensor_to_buffer",
      [](const nb::capsule& tensor,
         std::optional<nb_class_ptr<PyClient>> cpu_client,
         std::optional<nb_class_ptr<PyClient>> gpu_client) {
        return xla::ValueOrThrow(DLPackManagedTensorToBuffer(
            tensor, std::move(cpu_client), std::move(gpu_client)));
      },
      nb::arg("dlpack"), nb::arg("cpu_backend").none() = nb::none(),
      nb::arg("gpu_backend").none() = nb::none());
  m_nb.def("cuda_array_interface_to_buffer",
           xla::ValueOrThrowWrapper(CudaArrayInterfaceToBuffer), nb::arg("cai"),
           nb::arg("gpu_backend").none() = nb::none(),
           nb::arg("device_id").none() = nb::none());

  BuildIfrtProgramsSubmodule(m_nb);
  BuildProfilerSubmodule(m_nb);
  BuildOpsSubmodule(m_nb);
  BuildOutfeedReceiverSubmodule(m_nb);
  BuildPytreeSubmodule(m_nb);
  jax::BuildJaxjitSubmodule(m_nb);
  jax::BuildPmapSubmodule(m_nb);
  jax::BuildPjitSubmodule(m_nb);
  jax::BuildTransferGuardSubmodule(m_nb);
  BuildTracebackSubmodule(m_nb);
  BuildMlirSubmodule(m_nb);
  BuildCustomCallShardingPybindAPI(m_nb);

  // The following uses python bindings for PyClient defined above using
  // pybind11, and hence needs pybind11::module_ (not just nanobind::module_).
  xla::ifrt::proxy::BuildIfrtProxySubmodule(m_nb);

  nb::class_<tsl::PreemptionSyncManager> preemption_sync_manager(
      m_nb, "PreemptionSyncManager");
  preemption_sync_manager
      .def(
          "initialize",
          [](tsl::PreemptionSyncManager& manager,
             DistributedRuntimeClient* client) {
            tsl::CoordinationServiceAgent* agent =
                xla::ValueOrThrow(client->GetCoordinationServiceAgent());
            xla::ThrowIfError(manager.Initialize(agent));
          },
          nb::arg("distributed_client"))
      .def("reached_sync_point",
           [](tsl::PreemptionSyncManager& manager, int step_counter) {
             return manager.ReachedSyncPoint(step_counter);
           });
  m_nb.def("create_preemption_sync_manager",
           []() { return tsl::CreatePreemptionSyncManager(); });

  nb::class_<DistributedRuntimeService> distributed_runtime_service(
      m_nb, "DistributedRuntimeService");
  distributed_runtime_service.def("shutdown",
                                  &DistributedRuntimeService::Shutdown,
                                  nb::call_guard<nb::gil_scoped_release>());
  nb::class_<DistributedRuntimeClient> distributed_runtime_client(
      m_nb, "DistributedRuntimeClient");
  distributed_runtime_client
      .def("connect",
           [](DistributedRuntimeClient& self) {
             nb::gil_scoped_release gil_release;
             xla::ThrowIfError(self.Connect());
           })
      .def("shutdown",
           [](DistributedRuntimeClient& self) {
             nb::gil_scoped_release gil_release;
             xla::ThrowIfError(self.Shutdown());
           })
      // This method assumes that the value is a Python string. Use
      // `blocking_key_value_get_bytes()` if key_value_set() was called with a
      // Python bytes object as its value.
      .def(
          "blocking_key_value_get",
          [](DistributedRuntimeClient& client, std::string key,
             int64_t timeout_in_ms) {
            nb::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.BlockingKeyValueGet(
                key, absl::Milliseconds(timeout_in_ms)));
          },
          nb::arg("key"), nb::arg("timeout_in_ms"))
      // Same as `blocking_key_value_get()`, but retrieves the raw Python byte
      // values explicitly.
      .def(
          "blocking_key_value_get_bytes",
          [](DistributedRuntimeClient& client, std::string key,
             int64_t timeout_in_ms) -> nb::bytes {
            nb::gil_scoped_release gil_release;
            std::string result = xla::ValueOrThrow(client.BlockingKeyValueGet(
                key, absl::Milliseconds(timeout_in_ms)));
            return nb::bytes(result.data(), result.size());
          },
          nb::arg("key"), nb::arg("timeout_in_ms"))
      .def(
          "wait_at_barrier",
          [](DistributedRuntimeClient& client, std::string barrier_id,
             int64_t timeout_in_ms,
             std::optional<std::vector<int32_t>> process_ids) {
            nb::gil_scoped_release gil_release;
            xla::ThrowIfError(client.WaitAtBarrier(
                barrier_id, absl::Milliseconds(timeout_in_ms), process_ids));
          },
          nb::arg("barrier_id"), nb::arg("timeout_in_ms"),
          nb::arg("process_ids") = std::nullopt)
      // The key must be a string, but the value can either be a Python string
      // or bytes object.
      // With Python string values, use `key_value_set()` and
      // `blocking_key_value_get()`.
      // With Python byte object values, use `key_value_set()` and
      // `blocking_key_value_get_bytes()`.
      .def(
          "key_value_set",
          [](DistributedRuntimeClient& client, std::string_view key,
             std::string_view value, bool allow_overwrite) {
            nb::gil_scoped_release gil_release;
            xla::ThrowIfError(client.KeyValueSet(key, value, allow_overwrite));
          },
          nb::arg("key"), nb::arg("value"), nb::arg("allow_overwrite") = false)
      // The key must be a string, but the value must a
      // Python bytes object.
      // Use `key_value_set_bytes()` and `blocking_key_value_get_bytes()`.
      .def(
          "key_value_set_bytes",
          [](DistributedRuntimeClient& client, std::string_view key,
             nb::bytes value, bool allow_overwrite) {
            nb::gil_scoped_release gil_release;
            xla::ThrowIfError(client.KeyValueSet(
                key, std::string_view(value.c_str(), value.size()),
                allow_overwrite));
          },
          nb::arg("key"), nb::arg("value"), nb::arg("allow_overwrite") = false)
      // Assumes that all values in the directory are Python strings.
      .def(
          "key_value_dir_get",
          [](DistributedRuntimeClient& client, std::string_view key) {
            nb::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.KeyValueDirGet(key));
          },
          nb::arg("key"))
      // Assumes that all values in the directory are Python byte objects.
      // Same as `key_value_dir_get()`, but retrieves Python byte values
      // explicitly.
      .def(
          "key_value_dir_get_bytes",
          [](DistributedRuntimeClient& client, std::string_view key)
              -> std::vector<std::pair<std::string, nb::bytes>> {
            nb::gil_scoped_release gil_release;
            std::vector<std::pair<std::string, std::string>> result =
                xla::ValueOrThrow(client.KeyValueDirGet(key));
            // Convert std::string values to nb::bytes.
            std::vector<std::pair<std::string, nb::bytes>> kvs;
            kvs.reserve(result.size());
            for (const auto& kv : result) {
              kvs.push_back(std::pair(
                  kv.first, nb::bytes(kv.second.data(), kv.second.size())));
            }
            return kvs;
          },
          nb::arg("key"))
      .def(
          "key_value_delete",
          [](DistributedRuntimeClient& client, std::string_view key) {
            nb::gil_scoped_release gil_release;
            return xla::ThrowIfError(client.KeyValueDelete(key));
          },
          nb::arg("key"));

  m_nb.def(
      "get_distributed_runtime_service",
      [](std::string address, int num_nodes,
         std::optional<int> heartbeat_interval,
         std::optional<int> max_missing_heartbeats,
         std::optional<int> cluster_register_timeout,
         std::optional<int> shutdown_timeout)
          -> std::unique_ptr<DistributedRuntimeService> {
        CoordinationServiceImpl::Options options;
        options.num_nodes = num_nodes;
        if (heartbeat_interval.has_value()) {
          options.heartbeat_interval = absl::Seconds(*heartbeat_interval);
        }
        if (max_missing_heartbeats.has_value()) {
          options.max_missing_heartbeats = *max_missing_heartbeats;
        }
        if (cluster_register_timeout.has_value()) {
          options.cluster_register_timeout =
              absl::Seconds(*cluster_register_timeout);
        }
        if (shutdown_timeout.has_value()) {
          options.shutdown_timeout = absl::Seconds(*shutdown_timeout);
        }
        std::unique_ptr<DistributedRuntimeService> service =
            xla::ValueOrThrow(GetDistributedRuntimeService(address, options));
        return service;
      },
      nb::arg("address"), nb::arg("num_nodes"),
      nb::arg("heartbeat_interval").none() = std::nullopt,
      nb::arg("max_missing_heartbeats").none() = std::nullopt,
      nb::arg("cluster_register_timeout").none() = std::nullopt,
      nb::arg("shutdown_timeout").none() = std::nullopt);

  m_nb.def(
      "get_distributed_runtime_client",
      [](std::string address, int node_id, std::optional<int> rpc_timeout,
         std::optional<int> init_timeout, std::optional<int> shutdown_timeout,
         std::optional<int> heartbeat_interval,
         std::optional<int> max_missing_heartbeats,
         std::optional<std::function<void(absl::Status,
                                          bool coordinator_reported_failure)>>
             missed_heartbeat_callback,
         std::optional<bool> shutdown_on_destruction)
          -> std::shared_ptr<DistributedRuntimeClient> {
        DistributedRuntimeClient::Options options;
        options.node_id = node_id;
        if (rpc_timeout.has_value()) {
          options.rpc_timeout = absl::Seconds(*rpc_timeout);
        }
        if (init_timeout.has_value()) {
          options.init_timeout = absl::Seconds(*init_timeout);
        }
        if (shutdown_timeout.has_value()) {
          options.shutdown_timeout = absl::Seconds(*shutdown_timeout);
        }
        if (heartbeat_interval.has_value()) {
          options.heartbeat_interval = absl::Seconds(*heartbeat_interval);
        }
        if (max_missing_heartbeats.has_value()) {
          options.max_missing_heartbeats = *max_missing_heartbeats;
        }
        if (missed_heartbeat_callback.has_value()) {
          options.missed_heartbeat_callback =
              std::move(*missed_heartbeat_callback);
        }
        if (shutdown_on_destruction.has_value()) {
          options.shutdown_on_destruction = *shutdown_on_destruction;
        }
        return GetDistributedRuntimeClient(address, options);
      },
      nb::arg("address"), nb::arg("node_id"),
      nb::arg("rpc_timeout").none() = std::nullopt,
      nb::arg("init_timeout").none() = std::nullopt,
      nb::arg("shutdown_timeout").none() = std::nullopt,
      nb::arg("heartbeat_interval").none() = std::nullopt,
      nb::arg("max_missing_heartbeats").none() = std::nullopt,
      nb::arg("missed_heartbeat_callback").none() = std::nullopt,
      nb::arg("shutdown_on_destruction").none() = std::nullopt);

  m_nb.def("collect_garbage", []() { GlobalPyRefManager()->CollectGarbage(); });

  m_nb.def("is_optimized_build", &IsOptimizedBuild);

  m_nb.def("json_to_pprof_profile",
           xla::ValueOrThrowWrapper(JsonToPprofProfile),
           "Encodes the JSON representation of a pprof Profile into its binary "
           "protocol buffer encoding.");
  m_nb.def("pprof_profile_to_json",
           xla::ValueOrThrowWrapper(PprofProfileToJson),
           "Decodes an uncompressed pprof Profile protocol buffer into a JSON "
           "representation");

  RegisterCompileOnlyClient(m_nb);
  nb::class_<ifrt::Topology>(m_nb, "DeviceTopology")
      .def("_make_compile_only_devices",
           [](std::shared_ptr<ifrt::Topology> topology) {
             if (!llvm::isa<ifrt::PjRtTopology>(*topology)) {
               throw xla::XlaRuntimeError("Only PjRtTopologies are supported.");
             }
             return MakeCompileOnlyClient(
                        std::dynamic_pointer_cast<ifrt::PjRtTopology>(topology))
                 ->Devices();
           })
      .def_prop_ro(
          "platform",
          [](ifrt::Topology& topology) { return topology.platform_name(); })
      .def_prop_ro(
          "platform_version",
          [](ifrt::Topology& topology) { return topology.platform_version(); })
      .def("serialize",
           [](ifrt::Topology& topology) -> nb::bytes {
             std::string serialized = ValueOrThrow(topology.Serialize());
             return nb::bytes(serialized.data(), serialized.size());
           })
      .def("__getattr__",
           [](ifrt::Topology& topology, std::string_view name) -> nb::object {
             const auto& attrs = topology.Attributes();
             auto it = attrs.find(name);
             if (it != attrs.end()) {
               return std::visit([](auto&& v) { return nb::cast(v); },
                                 it->second);
             }
             throw nb::attribute_error(
                 absl::StrCat("Unknown attribute ", name).c_str());
           });

  nb::class_<ifrt::Executable>(m_nb, "Executable")
      .def("hlo_modules", ValueOrThrowWrapper(&ifrt::Executable::GetHloModules))
      .def("get_output_memory_kinds",
           xla::ValueOrThrowWrapper(&ifrt::Executable::GetOutputMemoryKinds))
      .def("get_output_shardings", &ifrt::Executable::GetOutputShardings)
      .def("get_parameter_layouts",
           ValueOrThrowWrapper(&ifrt::Executable::GetParameterLayouts))
      .def("get_output_layouts",
           xla::ValueOrThrowWrapper(&ifrt::Executable::GetOutputLayouts))
      .def("get_parameter_shardings", &ifrt::Executable::GetParameterShardings)
      .def("get_compiled_memory_stats",
           xla::ValueOrThrowWrapper(&ifrt::Executable::GetCompiledMemoryStats))
      .def("compile_options", &ifrt::Executable::GetCompileOptions)
      .def("serialize",
           [](const ifrt::Executable& exec) -> nb::bytes {
             std::string serialized = ValueOrThrow(exec.Serialize());
             return nb::bytes(serialized.data(), serialized.size());
           })
      .def("cost_analysis",
           xla::ValueOrThrowWrapper(&ifrt::Executable::GetCostAnalysis));

  m_nb.def("is_asan", IsAsan);
  m_nb.def("is_msan", IsMsan);
  m_nb.def("is_tsan", IsTsan);
  m_nb.def("is_sanitized", IsSanitized);

  m_nb.def(
      "batched_device_put",
      [](nb::object aval, nb::object sharding, std::vector<nb::object> xs,
         std::vector<const PyDevice*> dst_devices, bool committed,
         bool force_copy,
         PjRtClient::HostBufferSemantics host_buffer_semantics) -> nb::object {
        return ValueOrThrow(PyArray::BatchedDevicePut(
            nb::borrow(aval.ptr()), nb::borrow(sharding.ptr()), std::move(xs),
            std::move(dst_devices), committed, force_copy,
            host_buffer_semantics, jax::GetEnableX64()));
      },
      nb::arg("aval"), nb::arg("sharding"), nb::arg("xs"), nb::arg("devices"),
      nb::arg("committed") = true, nb::arg("force_copy") = false,
      nb::arg("host_buffer_semantics") =
          PjRtClient::HostBufferSemantics::kImmutableZeroCopy);

  m_nb.def("batched_block_until_ready", [](std::vector<nb::object> xs) {
    ThrowIfError(PyArray::BatchedBlockUntilReady(std::move(xs)));
  });

  m_nb.def("check_and_canonicalize_memory_kind",
           &jax::CheckAndCanonicalizeMemoryKind, nb::arg("memory_kind").none(),
           nb::arg("device_list"));
}  // NOLINT(readability/fn_size)

}  // namespace xla
