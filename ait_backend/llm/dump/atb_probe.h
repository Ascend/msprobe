/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef ATB_PROBE_H
#define ATB_PROBE_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <climits>


namespace atb {
const std::string ARGS_DUMP_TYPE_TENSOR = "tensors";
const std::string ARGS_DUMP_TYPE_CPU_PROFILING = "cpu_profiling";

constexpr int SAVE_TENSOR_BEFORE = 0;
constexpr int SAVE_TENSOR_AFTER = 1;
constexpr int SAVE_TENSOR_BOTH = 2;
constexpr int SAVE_TENSOR_DATA = 1;
constexpr int RANGE_COUNT = 2;
constexpr int SAVE_INTENSOR = 0;
constexpr int SAVE_OUTTENSOR = 1;
constexpr int SAVE_ALL_TENSOR = 2;

class Probe {
public:
    struct Tensor {
        std::string dype;
        std::string format;
        std::string shape;
        std::string path;
    };

    struct TensorInfo {
        std::string format;
        std::string dtype;
        std::string dims;
        uint8_t *hostData;
        uint64_t dataSize;
    };
public:
    static bool IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &optype);
    static bool IsSaveTensorData();
    static bool IsSaveTensorDesc();
    static bool IsSaveChild();
    static bool IsExecuteCountInRange(const uint64_t executeCount);
    static bool IsSaveTensorBefore();
    static bool IsSaveTensorAfter();
    static void SaveTensor(const std::string &format, const std::string &dtype,
        const std::string &dims, const void *hostData, uint64_t dataSize,
        const std::string &filePath);
    static void SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath);
    static bool IsSaveTiling();
    static bool IsSaveOuttensor();
    static bool IsSaveIntensor();
    static bool ReportOperationGraphEnable();
    static void ReportOperationGraph(const std::string &opName, const std::string &graph);
    static bool ReportOperationStatisticEnable();
    static void ReportOperationSetupStatistic(const uint64_t executeCount,
        const std::string &opname, const std::string &st);
    static void ReportOperationExecuteStatistic(const uint64_t executeCount,
        const std::string &opname, const std::string &st);
    static bool ReportOperationIOTensorEnable();
    static void ReportOperationIOTensor(const size_t executeCount, const std::string &opName,
        const std::string &opParam, const std::vector<atb::Probe::Tensor> &inTensors,
        const std::vector<atb::Probe::Tensor> &outTensors);
    static bool ReportKernelIOTensorEnable();
    static void ReportKernelIOTensor(const size_t executeCount, const std::string &opName,
        const std::string &opParam, const std::vector<atb::Probe::Tensor> &inTensors,
        const std::vector<atb::Probe::Tensor> &outTensors);
    static void SaveParam(const std::string &param, const std::string &filePath);
    static bool IsSaveParam();

    // ait llm antiCheck demo
    static bool IsOverflowCheck();
    static bool IsOverflowStop();
    static void ReportOverflowKernel(const std::string &kernelPath);
};
} // namespace atb

namespace atb_speed {

class SpeedProbe {
public:
    static bool IsReportModelTopoInfo(const std::string &modelName);
    static void ReportModelTopoInfo(const std::string &modelName, const std::string &graph);
};
} // namespace atb_speed

#endif