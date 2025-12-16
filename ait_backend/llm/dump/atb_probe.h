/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */


#ifndef ATB_PROBE_H
#define ATB_PROBE_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <climits>
#include <map>

#define EXPORT_LLM __attribute__ ((visibility("default")))

namespace atb {
const std::string TENSOR_AND_STATS_DATA_DIR = "data/";
constexpr int RANGE_COUNT = 2;

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
    EXPORT_LLM static void UpdateConfig();
    EXPORT_LLM static bool IsSaveTensorInSpecificDir(const std::string &tensorDir);
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

enum class TensorDType {
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
    TENSOR_DTYPE_BF16 = 27,
    TENSOR_DTYPE_INT4 = 29,
    TENSOR_DTYPE_UINT1 = 30,
    TENSOR_DTYPE_COMPLEX32 = 33,
    TENSOR_DTYPE_HIFLOAT8 = 34,
    TENSOR_DTYPE_FLOAT8_E5M2 = 35,
    TENSOR_DTYPE_FLOAT8_E4M3FN = 36,
    TENSOR_DTYPE_FLOAT8_E8M0 = 37,
    TENSOR_DTYPE_FLOAT6_E3M2 = 38,
    TENSOR_DTYPE_FLOAT6_E2M3 = 39,
    TENSOR_DTYPE_FLOAT4_E2M1 = 40,
    TENSOR_DTYPE_FLOAT4_E1M2 = 41,
};

enum class TensorFormat {
    TENSOR_FORMAT_UNDEFINED = -1,
    TENSOR_FORMAT_NCHW = 0,
    TENSOR_FORMAT_NHWC = 1,
    TENSOR_FORMAT_ND = 2,
    TENSOR_FORMAT_NC1HWC0 = 3,
    TENSOR_FORMAT_FRACTAL_Z = 4,
    TENSOR_FORMAT_NC1HWC0_C04 = 12,
    TENSOR_FORMAT_HWCN = 16,
    TENSOR_FORMAT_NDHWC = 27,
    TENSOR_FORMAT_FRACTAL_NZ = 29,
    TENSOR_FORMAT_NCDHW = 30,
    TENSOR_FORMAT_NDC1HWC0 = 32,
    TENSOR_FORMAT_FRACTAL_Z_3D = 33,
    TENSOR_FORMAT_NC = 35,
    TENSOR_FORMAT_NCL = 47,
    TENSOR_FORMAT_FRACTAL_NZ_C0_16 = 50,
    TENSOR_FORMAT_FRACTAL_NZ_C0_32 = 51,
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

const std::map<TensorDType, std::string> MAP_DTYPE_TO_STRING = {
    {TensorDType::TENSOR_DTYPE_FLOAT, "float"},
    {TensorDType::TENSOR_DTYPE_FLOAT16, "float16"},
    {TensorDType::TENSOR_DTYPE_INT8, "int8"},
    {TensorDType::TENSOR_DTYPE_INT32, "int32"},
    {TensorDType::TENSOR_DTYPE_UINT8, "uint8"},
    {TensorDType::TENSOR_DTYPE_INT16, "int16"},
    {TensorDType::TENSOR_DTYPE_UINT16, "uint16"},
    {TensorDType::TENSOR_DTYPE_UINT32, "uint32"},
    {TensorDType::TENSOR_DTYPE_INT64, "int64"},
    {TensorDType::TENSOR_DTYPE_UINT64, "uint64"},
    {TensorDType::TENSOR_DTYPE_DOUBLE, "double"},
    {TensorDType::TENSOR_DTYPE_BOOL, "bool"},
    {TensorDType::TENSOR_DTYPE_STRING, "string"},
    {TensorDType::TENSOR_DTYPE_COMPLEX64, "complex64"},
    {TensorDType::TENSOR_DTYPE_COMPLEX128, "complex128"},
    {TensorDType::TENSOR_DTYPE_BF16, "bf16"},
    {TensorDType::TENSOR_DTYPE_INT4, "int4"},
    {TensorDType::TENSOR_DTYPE_UINT1, "uint1"},
    {TensorDType::TENSOR_DTYPE_COMPLEX32, "complex32"},
    {TensorDType::TENSOR_DTYPE_HIFLOAT8, "hifloat8"},
    {TensorDType::TENSOR_DTYPE_FLOAT8_E5M2, "float8_e5m2"},
    {TensorDType::TENSOR_DTYPE_FLOAT8_E4M3FN, "float8_e4m3fn"},
    {TensorDType::TENSOR_DTYPE_FLOAT8_E8M0, "float8_e8m0"},
    {TensorDType::TENSOR_DTYPE_FLOAT6_E3M2, "float6_e3m2"},
    {TensorDType::TENSOR_DTYPE_FLOAT6_E2M3, "float6_e2m3"},
    {TensorDType::TENSOR_DTYPE_FLOAT4_E2M1, "float4_e2m1"},
    {TensorDType::TENSOR_DTYPE_FLOAT4_E1M2, "float4_e1m2"},
};

const std::map<TensorFormat, std::string> MAP_FORMAT_TO_STRING = {
    {TensorFormat::TENSOR_FORMAT_NCHW, "nchw"},
    {TensorFormat::TENSOR_FORMAT_NHWC, "nhwc"},
    {TensorFormat::TENSOR_FORMAT_ND, "nd"},
    {TensorFormat::TENSOR_FORMAT_NC1HWC0, "nc1hwc0"},
    {TensorFormat::TENSOR_FORMAT_FRACTAL_Z, "fractal_z"},
    {TensorFormat::TENSOR_FORMAT_NC1HWC0_C04, "nc1hwc0_c04"},
    {TensorFormat::TENSOR_FORMAT_HWCN, "hwcn"},
    {TensorFormat::TENSOR_FORMAT_NDHWC, "ndhwc"},
    {TensorFormat::TENSOR_FORMAT_FRACTAL_NZ, "fractal_nz"},
    {TensorFormat::TENSOR_FORMAT_NCDHW, "ncdhw"},
    {TensorFormat::TENSOR_FORMAT_NDC1HWC0, "ndc1hwc0"},
    {TensorFormat::TENSOR_FORMAT_FRACTAL_Z_3D, "fractal_z_3d"},
    {TensorFormat::TENSOR_FORMAT_NC, "nc"},
    {TensorFormat::TENSOR_FORMAT_NCL, "ncl"},
    {TensorFormat::TENSOR_FORMAT_FRACTAL_NZ_C0_16, "fractal_nz_c0_16"},
    {TensorFormat::TENSOR_FORMAT_FRACTAL_NZ_C0_32, "fractal_nz_c0_32"},
};

size_t GetTensorElementSize(const TensorDType dType);
TensorDType ConvertToTensorDType(int dType);
TensorFormat ConvertToTensorFormat(int format);
const std::string &GetDTypeStr(const TensorDType &dType);
const std::string &GetFormatStr(const TensorFormat &format);

// 工具函数接口的补充, 用于UT测试
float ConvertToFloat32(uint16_t value, size_t exponentBits, size_t mantissaBits);
} // namespace MKi

#endif
