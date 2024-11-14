/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
// This file contains nccl api for mesha to use.
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MESHA_NCCL_WRAPPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MESHA_NCCL_WRAPPER_H_
// Common place for all collective thunks to include nccl/rccl headers.
#if TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#include "third_party/nccl/nccl.h"
#endif
#include "pybind11/pybind11.h"  // from @pybind11
#include "xla/python/py_buffer.h"
#include "xla/python/py_client.h"
#include "xla/service/gpu/mesha_nccl_group_base.h"
namespace xla {
namespace gpu {
namespace mesha {
using MeshaNcclUid = std::vector<int8_t>;
using MeshaUuids = std::vector<int>;
class PyCommGroup : public CommGroup {
 public:
  PyCommGroup(std::shared_ptr<PyClient> backend);
  // Communication operations:
  Status NcclLocalAllGather(const MeshaNcclUid &key,
                            std::vector<pybind11::handle> buffers,
                            std::vector<uint> local_start_positions,
                            uint global_start, uint n_elements,
                            bool use_default_stream);
  Status NcclBroadcastPartialGPUs(const MeshaNcclUid &key,
                                  std::vector<pybind11::handle> buffers,
                                  std::vector<uint> local_start_positions,
                                  uint n_elements, int root_rank,
                                  bool use_recv_stream,
                                  bool use_default_stream);
  Status NcclSend(const MeshaNcclUid &key, pybind11::handle buffer, uint start,
                  uint n_elements, int peer_p2p_rank, bool use_default_stream);
  Status NcclRecv(const MeshaNcclUid &key, pybind11::handle buffer, uint start,
                  uint n_elements, int peer_p2p_rank, bool use_default_stream);
  // Sync functions:
  Status CommunicatorRecordEvents(const MeshaUuids &uuids, int num_devices,
                                  bool is_send);
  Status CommunicatorWaitEvents(const MeshaUuids &uuids, int num_devices,
                                bool is_send);
  void CommWaitCompute(bool is_send, bool is_compute, int device_id);
  void ComputeWaitComm(bool is_send, bool is_compute, int device_id);
};
// Cross Mesh Communication related
void SetPyCommGroup(std::string key, std::shared_ptr<PyCommGroup> g,
                    const MeshaNcclUid &uid);
// We add them here rather than Mesha_events to avoid circular deps in Bazel:
// Mesha_events > done_event_thunk > executable > client >
// ComputationWaitEvents
Status ComputationWaitEvents(const MeshaUuids &uuids,
                             std::shared_ptr<PyClient> client);
// Event context management
void ResetEventContext(std::shared_ptr<PyClient> client);
// Other functions
StatusOr<int> GetBufferDeviceId(pybind11::handle buffer);
}  // namespace mesha
}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MESHA_NCCL_WRAPPER_H_