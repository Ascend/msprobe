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
#include <map>

#define EXPORT_LLM __attribute__ ((visibility("default")))

namespace atb {
const std::string ARGS_DUMP_TYPE_TENSOR = "tensors";
const std::string ARGS_DUMP_TYPE_CPU_PROFILING = "cpu_profiling";

constexpr int SAVE_TENSOR_BEFORE = 0;
constexpr int SAVE_TENSOR_AFTER = 1;
constexpr int SAVE_TENSOR_BOTH = 2;
constexpr int SAVE_TENSOR_IN_BEFORE_OUT_AFTER = 1;
constexpr int SAVE_TENSOR_DATA = 1;
constexpr int SAVE_TENSOR_STATS = 1;
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
    EXPORT_LLM static bool IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &optype);
    EXPORT_LLM static bool IsSaveTensorData();
    EXPORT_LLM static bool IsSaveTensorDesc();
    EXPORT_LLM static bool IsSaveChild();
    EXPORT_LLM static bool IsExecuteCountInRange(const uint64_t executeCount);
    EXPORT_LLM static bool IsSaveTensorBefore();
    EXPORT_LLM static bool IsSaveTensorAfter();
    EXPORT_LLM static void SaveTensor(const std::string &format, const std::string &dtype,
        const std::string &dims, const void *hostData, uint64_t dataSize,
        const std::string &filePath);
    EXPORT_LLM static void SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath);
    EXPORT_LLM static bool IsSaveTiling();
    EXPORT_LLM static bool IsSaveOuttensor();
    EXPORT_LLM static bool IsSaveIntensor();
    EXPORT_LLM static bool ReportOperationGraphEnable();
    EXPORT_LLM static void ReportOperationGraph(const std::string &opName, const std::string &graph);
    EXPORT_LLM static bool ReportOperationStatisticEnable();
    EXPORT_LLM static void ReportOperationSetupStatistic(const uint64_t executeCount,
        const std::string &opname, const std::string &st);
    EXPORT_LLM static void ReportOperationExecuteStatistic(const uint64_t executeCount,
        const std::string &opname, const std::string &st);
    EXPORT_LLM static bool ReportOperationIOTensorEnable();
    EXPORT_LLM static void ReportOperationIOTensor(const size_t executeCount, const std::string &opName,
        const std::string &opParam, const std::vector<atb::Probe::Tensor> &inTensors,
        const std::vector<atb::Probe::Tensor> &outTensors);
    EXPORT_LLM static bool ReportKernelIOTensorEnable();
    EXPORT_LLM static void ReportKernelIOTensor(const size_t executeCount, const std::string &opName,
        const std::string &opParam, const std::vector<atb::Probe::Tensor> &inTensors,
        const std::vector<atb::Probe::Tensor> &outTensors);
    EXPORT_LLM static void SaveParam(const std::string &param, const std::string &filePath);
    EXPORT_LLM static bool IsSaveParam();

    // ait llm antiCheck demo
    EXPORT_LLM static bool IsOverflowCheck();
    EXPORT_LLM static bool IsOverflowStop();
    EXPORT_LLM static void ReportOverflowKernel(const std::string &kernelPath);
};
} // namespace atb

namespace atb_speed {

class SpeedProbe {
public:
    EXPORT_LLM static bool IsReportModelTopoInfo(const std::string &modelName);
    EXPORT_LLM static void ReportModelTopoInfo(const std::string &modelName, const std::string &graph);
};
} // namespace atb_speed

namespace Mki {

enum TensorDType : int {
    TENSOR_DTYPE_UNDEFINED = -1,
    TENSOR_DTYPE_FLOAT = 0,
    TENSOR_DTYPE_FLOAT16 = 1,
    TENSOR_DTYPE_INT8 = 2,
    TENSOR_DTYPE_INT32 = 3,
    TENSOR_DTYPE_UINT8 = 4,
    TENSOR_DTYPE_INT16 = 6,
    TENSOR_DTYPE_UINT16 = 7,
    TENSOR_DTYPE_UINT32 = 8,
    TENSOR_DTYPE_INT64 = 9,
    TENSOR_DTYPE_UINT64 = 10,
    TENSOR_DTYPE_DOUBLE = 11,
    TENSOR_DTYPE_BOOL = 12,
    TENSOR_DTYPE_STRING = 13,
    TENSOR_DTYPE_COMPLEX64 = 16,
    TENSOR_DTYPE_COMPLEX128 = 17,
    TENSOR_DTYPE_BF16 = 27
};

constexpr size_t HALF_DATA_SIZE = 2;
const std::string UNDEFINED_STR = "undefined";

const std::map<TensorDType, size_t> MAP_OF_DTYPE_SIZE = {
    {TensorDType::TENSOR_DTYPE_UNDEFINED, 0},
    {TensorDType::TENSOR_DTYPE_FLOAT, sizeof(float)},
    {TensorDType::TENSOR_DTYPE_FLOAT16, HALF_DATA_SIZE},
    {TensorDType::TENSOR_DTYPE_INT8, sizeof(int8_t)},
    {TensorDType::TENSOR_DTYPE_INT32, sizeof(int32_t)},
    {TensorDType::TENSOR_DTYPE_UINT8, sizeof(uint8_t)},
    {TensorDType::TENSOR_DTYPE_INT16, sizeof(int16_t)},
    {TensorDType::TENSOR_DTYPE_UINT16, sizeof(uint16_t)},
    {TensorDType::TENSOR_DTYPE_UINT32, sizeof(uint32_t)},
    {TensorDType::TENSOR_DTYPE_INT64, sizeof(int64_t)},
    {TensorDType::TENSOR_DTYPE_UINT64, sizeof(uint64_t)},
    {TensorDType::TENSOR_DTYPE_DOUBLE, sizeof(double)},
    {TensorDType::TENSOR_DTYPE_BOOL, sizeof(bool)},
    {TensorDType::TENSOR_DTYPE_BF16, HALF_DATA_SIZE},
    {TensorDType::TENSOR_DTYPE_COMPLEX64, sizeof(double)}
};

const std::map<std::string, TensorDType> MAP_STRING_TO_DTYPE = {
    { "float", TensorDType::TENSOR_DTYPE_FLOAT },
    { "float16", TensorDType::TENSOR_DTYPE_FLOAT16 },
    { "int8", TensorDType::TENSOR_DTYPE_INT8 },
    { "int32", TensorDType::TENSOR_DTYPE_INT32 },
    { "uint8", TensorDType::TENSOR_DTYPE_UINT8 },
    { "int16", TensorDType::TENSOR_DTYPE_INT16 },
    { "uint16", TensorDType::TENSOR_DTYPE_UINT16 },
    { "uint32", TensorDType::TENSOR_DTYPE_UINT32 },
    { "int64", TensorDType::TENSOR_DTYPE_INT64 },
    { "uint64", TensorDType::TENSOR_DTYPE_UINT64 },
    { "double", TensorDType::TENSOR_DTYPE_DOUBLE },
    { "bool", TensorDType::TENSOR_DTYPE_BOOL },
    { "string", TensorDType::TENSOR_DTYPE_STRING },
    { "complex64", TensorDType::TENSOR_DTYPE_COMPLEX64 },
    { "complex128", TensorDType::TENSOR_DTYPE_COMPLEX128 },
    { "bf16", TensorDType::TENSOR_DTYPE_BF16 },
};

const std::map<int, std::string> MAP_DTYPE_TO_STRING = {
    { TensorDType::TENSOR_DTYPE_FLOAT, "float" },
    { TensorDType::TENSOR_DTYPE_FLOAT16, "float16" },
    { TensorDType::TENSOR_DTYPE_INT8, "int8" },
    { TensorDType::TENSOR_DTYPE_INT32, "int32" },
    { TensorDType::TENSOR_DTYPE_UINT8, "uint8" },
    { TensorDType::TENSOR_DTYPE_INT16, "int16" },
    { TensorDType::TENSOR_DTYPE_UINT16, "uint16" },
    { TensorDType::TENSOR_DTYPE_UINT32, "uint32" },
    { TensorDType::TENSOR_DTYPE_INT64, "int64" },
    { TensorDType::TENSOR_DTYPE_UINT64, "uint64" },
    { TensorDType::TENSOR_DTYPE_DOUBLE, "double" },
    { TensorDType::TENSOR_DTYPE_BOOL, "bool" },
    { TensorDType::TENSOR_DTYPE_STRING, "string" },
    { TensorDType::TENSOR_DTYPE_COMPLEX64, "complex64" },
    { TensorDType::TENSOR_DTYPE_COMPLEX128, "complex128" },
    { TensorDType::TENSOR_DTYPE_BF16, "bf16" },
};

size_t GetTensorElementSize(const TensorDType dtype);
TensorDType GetDTypeWithStr(const std::string &typeStr);
const std::string &GetStrWithDType(int dType);

// 补充
float ConvertToFloat32(uint16_t value, size_t exponentBits, size_t mantissaBits);
} // namespace MKi

#endif