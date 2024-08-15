/* Copyright 2022 The OpenXLA Authors.

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

#include <iostream>
#include <ostream>
#include <string>

#include "absl/status/status.h"
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
