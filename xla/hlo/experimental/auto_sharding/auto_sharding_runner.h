#include <iostream>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "xla/xla_data.pb.h"
#include "xla/status.h"
#include "xla/tools/hlo_module_loader.h"
#include "tsl/platform/init_main.h"
#include "xla/pjrt/pjrt_executable.h"


namespace xla {
namespace spmd {
// namespace {

absl::Status RunAutoShardingPassFromFile(const std::string& file_name);

// TODO(yonghao): Check correctness of compile options and modules
Status PreCompileCheck(const CompileOptions& options);

StatusOr<HloModuleConfig> CreateHloModuleConfig(const HloModule* hlo_module,
                                                const CompileOptions options);


Status RunAutoShardingPass(HloModule* hlo_module,
                           const CompileOptions& options);

Status RunSpmdPartitionerPass(HloModule* hlo_module,
                              const CompileOptions& options);

Status SetHloModuleOutputShardings(HloModule* module,
                                   const std::vector<OpSharding>& op_shardings);

Status SetHloModuleInputShardings(HloModule* module,
                                  const std::vector<OpSharding>& op_shardings);

// }  // namespace
}  // namespace spmd
}  // namespace xla