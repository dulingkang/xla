// This file contains nccl api for mesha to use.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MESHA_NCCL_GROUP_BASE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MESHA_NCCL_GROUP_BASE_H_
// Common place for all collective thunks to include nccl/rccl headers.
#if TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#include "third_party/nccl/nccl.h"
#endif

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/gpu/nccl_utils.h"
#include "xla/service/rendezvous.h"

namespace xla {
namespace gpu {
namespace mesha {
using MeshaNcclUid = std::vector<int8_t>;
using MeshaUuids = std::vector<int>;

class CommGroup {
 public:
  CommGroup(PjRtStreamExecutorClient *client);
  // Communicator related functions:
  Status NcclCreateCommunicators(int world_size,
                                 const std::vector<int> &device_global_ranks,
                                 const std::vector<int> &device_ids,
                                 const MeshaNcclUid &nccl_uid_vec);

  Status NcclDestroyComms(const MeshaNcclUid &storage);

  // Communication operations:
  Status NcclLocalAllGatherImpl(const MeshaNcclUid &key,
                                std::vector<PjRtBuffer *> buffers,
                                std::vector<uint> local_start_positions,
                                uint global_start, uint n_elements,
                                bool use_default_stream);

  Status NcclBroadcastPartialGPUsImpl(const MeshaNcclUid &key,
                                      std::vector<PjRtBuffer *> buffers,
                                      std::vector<uint> local_start_positions,
                                      uint n_elements, int root_rank,
                                      bool use_recv_stream,
                                      bool use_default_stream);

  Status NcclSendImpl(const MeshaNcclUid &key, PjRtBuffer *buffer, uint start,
                      uint n_elements, int peer_p2p_rank,
                      bool use_default_stream);

  Status NcclRecvImpl(const MeshaNcclUid &key, PjRtBuffer *buffer, uint start,
                      uint n_elements, int peer_p2p_rank,
                      bool use_default_stream);

  // Other functions
  NcclComm::Lock AcquireComm(const MeshaNcclUid &uuids, int device_id);

 protected:
  std::vector<std::unique_ptr<se::Stream>> send_streams, recv_streams;
  absl::flat_hash_map<MeshaNcclUid, std::vector<int>> local_ids;
  std::vector<se::StreamExecutor *> executors;
  PjRtStreamExecutorClient *client_;

 private:
  ThreadSafeMap<std::pair<MeshaNcclUid, int>, NcclComm> comm_map;
};

// Cross-mesh allreduce thunk related
void SetCommGroup(std::string key, std::shared_ptr<CommGroup> g,
                  const MeshaNcclUid &uid);

NcclComm::Lock GetCommunicator(std::string key, size_t device_id);

// Other functions
ncclUniqueId NcclUidDeserialize(const MeshaNcclUid &nccl_uid_chars);

StatusOr<MeshaNcclUid> NcclGetUniqueId();

Status NcclCreateCommunicators2(int world_size,
                                 const std::vector<int> &device_global_ranks,
                                 const std::vector<int> &device_ids,
                                 const MeshaNcclUid &nccl_uid_vec);

StatusOr<int> NcclGetVersion();
}  // namespace mesha
}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MESHA_NCCL_GROUP_BASE_H_
