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

#include "atb_probe.h"

#include <dlfcn.h>
#include <syscall.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
// for calculating dumped-tensor statistics
#include <algorithm>
#include <chrono>
#include <complex>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "Statistics.h"
// endfor
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include "DumpThreadPool.h"
#include "ait_logger.h"
#include "bin_file.h"
#include "const.h"
#include "file.h"
#include "nlohmann/json.hpp"
#include "safety_guard.h"
#include "utils.h"

using ordered_json = nlohmann::ordered_json;
using MsConst::SAFETY_RET;

// C++11标准不支持make_unique函数，此处为make_unique的简单实现
#if __cplusplus < 201402L
namespace std
{
// 非数组类型
template <typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type make_unique(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// 数组类型(T[])
template <typename T>
using EnableIfDynamicArray =
    typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0, std::unique_ptr<T>>::type;

template <typename T>
EnableIfDynamicArray<T> make_unique(size_t size)
{
    if (size <= 0)
    {
        throw std::invalid_argument("make_unique: array size must be greater than 0");
    }
    using U = typename std::remove_extent<T>::type;
    return std::unique_ptr<T>(new U[size]);
}

// 禁用定长数组(如T[5])
template <typename T, typename... Args>
typename std::enable_if<std::extent<T>::value != 0, std::unique_ptr<T>>::type make_unique(Args &&...) = delete;
}  // namespace std
#endif

namespace
{
unsigned long long g_minDiskSpaceFreeSize = 2147483648;  // 2G
static const int POOLNUM = 12;
constexpr size_t FREE_SIZE_MULTIPLE_OF_DATA_SIZE = 2;  // free size至少两倍data size大小
constexpr uint32_t MAX_RECUR_DEPTH = 1000;
constexpr uint32_t MAX_PATH_LENGTH = MsConst::FULL_PATH_LENGTH_MAX;
constexpr const char *LIBASCENDCL_SO = "libascendcl.so";
constexpr const char *ASCEND_TOOLKIT_HOME = "ASCEND_TOOLKIT_HOME";

template <typename T>
inline bool Likely(T &&condition) noexcept
{
    return __builtin_expect(static_cast<bool>(condition), 1);
}

template <typename T>
inline bool Unlikely(T &&condition) noexcept
{
    return __builtin_expect(static_cast<bool>(condition), 0);
}

static int SafetyStoi(const char *env, int defaultValue)
{
    int ans = defaultValue;
    try
    {
        ans = std::stoi(env);
    }
    catch (const std::invalid_argument &e)
    {
        AIT_LOG_ERROR("The argument passed in can not be converted to int.");
        return ans;
    }
    catch (const std::out_of_range &e)
    {
        AIT_LOG_ERROR("The value of argument is out of range.");
        return ans;
    }
    return ans;
}

struct LayerGraphMap
{
    std::map<std::string, std::string> layerGraphMap_;

    void SaveLayerGraph(const std::string &opName, const std::string &graph) { layerGraphMap_[opName] = graph; };

    std::string GetLayerGraph(const std::string &opName)
    {
        auto it = layerGraphMap_.find(opName);
        return (it == layerGraphMap_.end()) ? "" : it->second;
    };
};

std::unique_ptr<ThreadPool::DumpThreadPool> &GetGlobalPool(size_t numThreads = POOLNUM)
{
    static std::unique_ptr<ThreadPool::DumpThreadPool> instance =
        std::make_unique<ThreadPool::DumpThreadPool>(numThreads);
    return instance;
}

template <typename T>
bool ParseJsonBaseObj2Var(const nlohmann::json &content, const std::string &field, T &output)
{
    nlohmann::json::const_iterator iter = content.find(field);
    if (iter == content.end())
    {
        return false;
    }
    try
    {
        output = iter->get<T>();
        return true;
    }
    catch (const nlohmann::detail::type_error &e)
    {
        return false;
    }
}
}  // namespace

static int GetFreeSpace(std::string path, unsigned long long &freeSpace)
{
    struct statvfs diskInfo;

    if (statvfs(path.c_str(), &diskInfo) == -1)
    {
        AIT_LOG_ERROR("statvfs() error:" + Utils::GetLastErrorStr());
        return 1;
    }
    freeSpace = diskInfo.f_bavail * diskInfo.f_bsize;
    return 0;
}

static int32_t GetCurrentDeviceId()
{
    int32_t deviceId = -1;
    const char *ascendToolkitHome = std::getenv(ASCEND_TOOLKIT_HOME);
    if (ascendToolkitHome == nullptr)
    {
        return deviceId;
    }
    std::string ascendclPath = std::string(ascendToolkitHome) + "/lib64/" + LIBASCENDCL_SO;
    if (ascendclPath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of ascendclPath must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                      " characters, but got " + std::to_string(ascendclPath.length()));
        return deviceId;
    }
    struct stat fileStat;
    if (stat(ascendclPath.c_str(), &fileStat) != 0)
    {
        return deviceId;
    }
    mode_t permissions = fileStat.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    if ((permissions & MsConst::READ_FILE_NOT_PERMITTED) > 0)
    {
        return deviceId;
    }
    void *handle = dlopen(ascendclPath.c_str(), RTLD_LAZY);
    if (handle == nullptr)
    {
        return deviceId;
    }
    int (*getDeviceFunc)(int32_t *) = reinterpret_cast<int (*)(int32_t *)>(dlsym(handle, "aclrtGetDevice"));
    if (getDeviceFunc == nullptr)
    {
        dlclose(handle);
        return deviceId;
    }
    int aclRet = getDeviceFunc(&deviceId);
    if (aclRet != 0)
    {
        deviceId = -1;
    }
    dlclose(handle);
    return deviceId;
}

static int32_t GetCurrentProcessId()
{
    int32_t pid = getpid();
    if (pid == -1)
    {
        AIT_LOG_ERROR("get pid failed");
    }
    return pid;
}

static bool IsPrefix(const std::string &str, const std::string &prefix)
{
    return str.compare(0, prefix.length(), prefix) == 0;
}

static const std::string &GetOutDir()
{
    static std::string outDir = "";
    if (outDir == "")
    {
        const char *outputDir = std::getenv("ATB_OUTPUT_DIR");
        outDir = (outputDir != nullptr ? outputDir : "./");
        outDir = GetRealPath(outDir);
        while (!outDir.empty() && outDir.back() == '/')
        {
            outDir.pop_back();
        }
        outDir = outDir + "/atb_dump_data/";
        if (outDir.length() > MAX_PATH_LENGTH)
        {
            AIT_LOG_ERROR("The path length of outDir must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                          " characters, but got " + std::to_string(outDir.length()));
            outDir = "";
            return outDir;
        }
        bool ret = Utils::CheckDirectory(outDir);
        if (!ret)
        {
            AIT_LOG_ERROR("Create directory failed: " + outDir);
            outDir = "";
        }
    }
    return outDir;
}

static bool IsInTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind('/');
    std::string fileName = filePath;
    if (sepPos != std::string::npos)
    {
        fileName.erase(0, sepPos + 1U);
    }
    bool flag = (fileName.find("intensor") != std::string::npos) || (fileName.find("inTensor") != std::string::npos);
    AIT_LOG_DEBUG("IsInTensorBinPath: " + std::to_string(flag));
    return flag;
}

static bool IsOutTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind('/');
    std::string fileName = filePath;
    if (sepPos != std::string::npos)
    {
        fileName.erase(0, sepPos + 1U);
    }
    bool flag = (fileName.find("outtensor") != std::string::npos) || (fileName.find("outTensor") != std::string::npos);
    AIT_LOG_DEBUG("IsOutTensorBinPath: " + std::to_string(flag));
    return flag;
}

static void DfsToModifyGraphTensors(ordered_json &curNodeToSave,
                                    const std::vector<std::string> &fatherNodeTensorNameList,
                                    const ordered_json &curNodeInput, uint32_t curDepth = 1)
{
    if (curDepth >= MAX_RECUR_DEPTH)
    {
        AIT_LOG_ERROR("The depth of the current graph structure is too deep, and the traversal node fails");
        throw std::runtime_error("The maximum recursive depth exceeds " + std::to_string(MAX_RECUR_DEPTH) + ".");
    }
    std::string opName = curNodeInput["opName"].get<std::string>();
    curNodeToSave["opName"] = curNodeInput["opName"];
    curNodeToSave["opType"] = curNodeInput["opType"];
    curNodeToSave["param"] = curNodeInput["param"];

    std::vector<std::string> curNodeTensorNameList;

    // 子节点的inTensors\outTensors为父节点的fatherNodeTensorNameList子集, 根据inTensorIds、outTensorIds获取
    for (auto item : curNodeInput["inTensorIds"])
    {
        uint32_t inputIndex = item.get<uint32_t>();
        if (inputIndex >= fatherNodeTensorNameList.size())
        {
            AIT_LOG_ERROR("inputIndex out of fatherNodeTensorNameList: " + opName);
            return;
        }
        curNodeToSave["inTensors"].emplace_back(fatherNodeTensorNameList[inputIndex]);
        curNodeTensorNameList.emplace_back(fatherNodeTensorNameList[inputIndex]);
    }

    for (auto item : curNodeInput["outTensorIds"])
    {
        uint32_t outputIndex = item.get<uint32_t>();
        if (outputIndex >= fatherNodeTensorNameList.size())
        {
            AIT_LOG_ERROR("outputIndex out of fatherNodeTensorNameList: " + opName);
            return;
        }
        curNodeToSave["outTensors"].emplace_back(fatherNodeTensorNameList[outputIndex]);
        curNodeTensorNameList.emplace_back(fatherNodeTensorNameList[outputIndex]);
    }

    // 子节点的internalTensors根据自己的opName + id, 组成tensor name
    uint32_t internalTensorNum = (curNodeInput.find("internalTensorNum") == curNodeInput.end())
                                     ? 0
                                     : curNodeInput["internalTensorNum"].get<uint32_t>();
    for (size_t i = 0; i < internalTensorNum; i++)
    {
        std::string tensorName = opName + "_internal_" + std::to_string(i);
        curNodeToSave["internalTensors"].emplace_back(tensorName);
        curNodeTensorNameList.emplace_back(tensorName);
    }

    // 递归调用获取子节点信息
    if (curNodeInput.find("nodes") != curNodeInput.end())
    {
        for (auto childNodeInput : curNodeInput["nodes"])
        {
            ordered_json childNodeToSave;
            DfsToModifyGraphTensors(childNodeToSave, curNodeTensorNameList, childNodeInput, curDepth + 1);
            curNodeToSave["nodes"].emplace_back(childNodeToSave);
        }
    }
    return;
}

static LayerGraphMap g_layerGraphMap;
static unsigned long long g_aitOperationBaseId(0);
static void MergeLayerTopoInfo(ordered_json &layerJson)
{
    // 获取atb仓打桩保存的layer的拓扑信息
    layerJson["opType"] = layerJson["opName"];
    std::string opName = layerJson["opName"].get<std::string>() + "_" + std::to_string(g_aitOperationBaseId++);
    layerJson["opName"] = opName;
    std::string atbLayerGraph = g_layerGraphMap.GetLayerGraph(opName);
    if (atbLayerGraph == "")
    {
        return;
    }
    ordered_json atbLayerJson;
    // inTensor和outTensor从model里获取，internalTensor自己申请id, 组成layerTensorNameList
    try
    {
        atbLayerJson = ordered_json::parse(atbLayerGraph);
    }
    catch (const ordered_json::parse_error &ex)
    {
        AIT_LOG_ERROR("json parse error! opName:" + opName);
        AIT_LOG_ERROR("message: " + std::string(ex.what()) + '\n' + "exception id: " + std::to_string(ex.id) + '\n' +
                      "byte position of error: " + std::to_string(ex.byte));
        return;
    }
    std::vector<std::string> layerTensorNameList;
    for (auto item : layerJson["inTensors"])
    {
        layerTensorNameList.emplace_back(item);
    }

    for (auto item : layerJson["outTensors"])
    {
        layerTensorNameList.emplace_back(item);
    }

    uint32_t internalTensorNum = (atbLayerJson.find("internalTensorNum") == atbLayerJson.end())
                                     ? 0
                                     : atbLayerJson["internalTensorNum"].get<uint32_t>();
    for (size_t i = 0; i < internalTensorNum; i++)
    {
        std::string tensorName = opName + "_internal_" + std::to_string(i);
        layerJson["internalTensors"].emplace_back(tensorName);
        layerTensorNameList.emplace_back(tensorName);
    }

    // 递归调用获取每个layer的子节点信息
    if (atbLayerJson.find("nodes") != atbLayerJson.end())
    {
        for (auto childNodeInput : atbLayerJson["nodes"])
        {
            ordered_json childNodeToSave;
            DfsToModifyGraphTensors(childNodeToSave, layerTensorNameList, childNodeInput);
            layerJson["nodes"].emplace_back(childNodeToSave);
        }
    }
    return;
}

namespace Mki
{

size_t GetTensorElementSize(const TensorDType dtype)
{
    auto iter = MAP_OF_DTYPE_SIZE.find(dtype);
    if (iter == MAP_OF_DTYPE_SIZE.end())
    {
        AIT_LOG_ERROR("Get Tensor ElementSize:dtype not found!");
        return 0;
    }
    return iter->second;
}

TensorDType ConvertToTensorDType(int dType)
{
    TensorDType tensorDType = TensorDType::TENSOR_DTYPE_UNDEFINED;
    switch (dType)
    {
        case static_cast<int>(TensorDType::TENSOR_DTYPE_UNDEFINED):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT16):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_INT8):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_INT32):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT8):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_INT16):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT16):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT32):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_INT64):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT64):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_DOUBLE):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_BOOL):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_STRING):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_COMPLEX64):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_COMPLEX128):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_BF16):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_INT4):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_UINT1):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_COMPLEX32):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_HIFLOAT8):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT8_E5M2):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT8_E4M3FN):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT8_E8M0):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT6_E3M2):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT6_E2M3):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT4_E2M1):
        case static_cast<int>(TensorDType::TENSOR_DTYPE_FLOAT4_E1M2):
            tensorDType = static_cast<TensorDType>(dType);
            break;
        default:
            break;
    }
    return tensorDType;
}

TensorFormat ConvertToTensorFormat(int format)
{
    TensorFormat tensorFormat = TensorFormat::TENSOR_FORMAT_UNDEFINED;
    switch (format)
    {
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_UNDEFINED):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NCHW):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NHWC):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_ND):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NC1HWC0):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_Z):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NC1HWC0_C04):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_HWCN):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NDHWC):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_NZ):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NCDHW):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NDC1HWC0):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_Z_3D):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NC):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_NCL):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_NZ_C0_16):
        case static_cast<int>(TensorFormat::TENSOR_FORMAT_FRACTAL_NZ_C0_32):
            tensorFormat = static_cast<TensorFormat>(format);
            break;
        default:
            break;
    }
    return tensorFormat;
}

const std::string &GetDTypeStr(const TensorDType &dType)
{
    auto it = MAP_DTYPE_TO_STRING.find(dType);
    if (it != MAP_DTYPE_TO_STRING.end())
    {
        return it->second;
    }
    return UNDEFINED_STR;
}

const std::string &GetFormatStr(const TensorFormat &format)
{
    auto it = MAP_FORMAT_TO_STRING.find(format);
    if (it != MAP_FORMAT_TO_STRING.end())
    {
        return it->second;
    }
    return UNDEFINED_STR;
}

float ConvertToFloat32(uint16_t value, size_t exponentBits, size_t mantissaBits)
{
    // Determine the bias of the semi-precision type
    int32_t exponentBias = (1 << (exponentBits - 1)) - 1;

    // Obtain the mask
    uint16_t exponentMask = ((1 << exponentBits) - 1) << mantissaBits;
    uint16_t mantissaMask = (1 << mantissaBits) - 1;
    uint16_t signMask = 1 << (exponentBits + mantissaBits);

    // Extract symbol bits
    int sign = (value & signMask) ? -1 : 1;

    // Extract index and mantissa
    int32_t rawExponent = (value & exponentMask) >> mantissaBits;
    uint32_t mantissa = value & mantissaMask;

    // Handle special values
    if (rawExponent == (1 << exponentBits) - 1)
    {  // All 1s represent NaN or infinity
        if (mantissa != 0)
        {
            return std::numeric_limits<float>::quiet_NaN();  // NaN
        }
        else
        {
            return sign * std::numeric_limits<float>::infinity();  // Infinity
        }
    }
    else if (rawExponent == 0)
    {  // Exponents with all zeros indicate non-normalized numbers or zeros
        if (mantissa == 0)
        {
            return sign * 0.0f;  // Zero
        }
        else
        {
            // Unnormalized number
            float result = sign * std::ldexp(static_cast<float>(mantissa),
                                             1 - static_cast<int>(exponentBias) - static_cast<int>(mantissaBits));
            return result;
        }
    }

    // Normalized number
    float normalizedMantissa = 1.0f + static_cast<float>(mantissa) / (1 << mantissaBits);
    float result = sign * std::ldexp(normalizedMantissa, rawExponent - exponentBias);
    return result;
}
}  // namespace Mki

namespace atb
{

constexpr size_t CONFIG_UPDATE_PERIOD = 16384;
constexpr size_t LAYER_PATH_DEPTH = 3;
constexpr int NO_FILTER_FOR_DATA = 0;
constexpr int FILTER_OPERATION_DATA = 1;
constexpr int FILTER_KERNEL_DATA = 2;
const std::string TENSOR_TASK = "tensor";
const std::string STATS_TASK = "statistics";
const std::string ALL_TASK = "all";
const std::string INTENSOR_PREFIX = "intensor";
const std::string OUTTENSOR_PREFIX = "outtensor";
const std::string STATS_FILE_NAME = "statistic.csv";
const std::vector<std::string> VALID_TASKS = {TENSOR_TASK, STATS_TASK, ALL_TASK};

struct BinFileInfo
{
    std::string format;
    std::string dtype;
    std::string dims;
    std::string filePath;
    const std::string opId;
    const std::string opName;
    const void *hostData;
    const uint64_t dataSize;
};

static size_t g_configUpdateTimes;

static bool g_isAtbRunning;
static bool g_isDebugMode;
static bool g_dumpEnable;
static bool g_saveChild;

static std::string g_configPath;
static std::string g_task;
static std::string g_executionCountRange;
static std::string g_operationId;
static std::string g_operationName;
static std::string g_device;

static int g_filterLevel;

static std::vector<std::string> g_splitOperationIds;
static std::vector<std::string> g_splitOperationNames;
static std::vector<std::string> g_splitDeviceIds;

static std::unordered_map<std::string, std::string> g_realTensors;

static std::set<std::string> g_DumpedLayerSet;

static bool OrderOperations(const ordered_json &operation, const std::string &idPrefix,
                            std::vector<std::string> &orderedOperations,
                            std::unordered_map<std::string, std::vector<std::string>> &opAndTensors, int &depth)
{
    depth += 1;
    if (depth > MAX_RECUR_DEPTH)
    {
        return false;
    }
    if (operation.find("opName") == operation.end())
    {
        return false;
    }

    try
    {
        std::string opName = operation.find("opName")->get<std::string>();
        size_t found = opName.find(idPrefix);
        if (found == std::string::npos)
        {
            return false;
        }

        std::string opId = opName.substr(found + 1);
        orderedOperations.emplace_back(opId);
        opAndTensors[opId + "_" + INTENSOR_PREFIX] = operation.find("inTensors")->get<std::vector<std::string>>();
        opAndTensors[opId + "_" + OUTTENSOR_PREFIX] = operation.find("outTensors")->get<std::vector<std::string>>();

        if (operation.find("nodes") == operation.end())
        {
            return true;
        }

        auto nodes = operation.find("nodes")->get<std::vector<ordered_json>>();
        for (const auto &node : nodes)
        {
            if (!OrderOperations(node, idPrefix, orderedOperations, opAndTensors, depth))
            {
                return false;
            }
        }
    }
    catch (const nlohmann::detail::type_error &e)
    {
        return false;
    }

    return true;
}

static void PreDealWhenPassingOpTensors(const std::string &operationId, std::stack<std::string> &orderedOperationId,
                                        std::vector<std::vector<std::string>> &orderedInputAndOutput)
{
    int loopCount = 0;
    while (!orderedOperationId.empty())
    {
        if (++loopCount > MAX_RECUR_DEPTH || operationId.find(orderedOperationId.top()) == 0)
        {
            return;
        }

        orderedInputAndOutput.emplace_back(std::vector<std::string>{orderedOperationId.top(), OUTTENSOR_PREFIX});
        orderedOperationId.pop();
    }
}

static bool NeedSave(const std::string &tensorName, const std::vector<std::string> &lastTensors,
                     const std::unordered_map<std::string, std::string> &tensorsAliases)
{
    if (tensorsAliases.find(tensorName) == tensorsAliases.end() ||
        std::find(lastTensors.begin(), lastTensors.end(), tensorName) != lastTensors.end())
    {
        return true;
    }

    return false;
}

static void FilterUnnecessaryData(const ordered_json &layerStructure)
{
    std::string layerName = layerStructure.find("opName")->get<std::string>();
    size_t found = layerName.rfind('_');
    if (found == std::string ::npos)
    {
        return;
    }
    std::string idPrefix = layerName.substr(found);

    std::vector<std::string> orderedOperations;
    std::unordered_map<std::string, std::vector<std::string>> opAndTensors;
    std::stack<std::string> orderedOperationId;
    std::vector<std::vector<std::string>> orderedInputAndOutput;
    int depth = 0;
    auto ret = OrderOperations(layerStructure, idPrefix, orderedOperations, opAndTensors, depth);
    if (!ret || orderedOperations.empty())
    {
        return;
    }

    std::string lastId;
    for (const auto &operationId : orderedOperations)
    {
        if (!lastId.empty() && operationId.find(lastId) != 0)
        {
            PreDealWhenPassingOpTensors(operationId, orderedOperationId, orderedInputAndOutput);
        }
        orderedInputAndOutput.emplace_back(std::vector<std::string>{operationId, INTENSOR_PREFIX});
        orderedOperationId.emplace(operationId);
        lastId = operationId;
    }

    while (!orderedOperationId.empty())
    {
        orderedInputAndOutput.emplace_back(std::vector<std::string>{orderedOperationId.top(), OUTTENSOR_PREFIX});
        orderedOperationId.pop();
    }

    std::unordered_map<std::string, std::string> tensorsAliases;
    for (size_t i = 0; i < orderedInputAndOutput.size(); i++)
    {
        const std::string ioType = orderedInputAndOutput.at(i).at(1);
        std::string opWithIoType = orderedInputAndOutput.at(i).at(0) + "_" + ioType;
        const std::vector<std::string> &tensors = opAndTensors[opWithIoType];
        std::vector<std::string> lastTensors;
        if (i != 0 && ioType == OUTTENSOR_PREFIX && orderedInputAndOutput.at(i - 1).at(1) == INTENSOR_PREFIX)
        {
            lastTensors = opAndTensors[orderedInputAndOutput.at(i - 1).at(0) + "_" + INTENSOR_PREFIX];
        }
        for (size_t j = 0; j < tensors.size(); j++)
        {
            if (NeedSave(tensors[j], lastTensors, tensorsAliases))
            {
                g_realTensors[opWithIoType + std::to_string(j) + ".bin"] = "bin";
                tensorsAliases[tensors.at(j)] = opWithIoType + std::to_string(j) + ".bin";
            }
            else
            {
                g_realTensors[opWithIoType + std::to_string(j) + ".bin"] = tensorsAliases.at(tensors.at(j));
            }
        }
    }
}

static std::vector<std::string> SplitDataPath(const std::string &dataPath)
{
    std::vector<std::string> splitPath = SplitString(dataPath.c_str(), '/');
    if (splitPath.size() < LAYER_PATH_DEPTH)
    {
        AIT_LOG_WARNING("Expected a valid data path whose depth must not be less than " +
                        std::to_string(LAYER_PATH_DEPTH) + " but got " + std::to_string(splitPath.size()));
        splitPath.clear();
    }

    return splitPath;
}

static std::string GetOpIdFromDataPath(const std::vector<std::string> &dataPathVec, size_t endOffset)
{
    std::string opId;
    if (dataPathVec.size() <= endOffset)
    {
        return opId;
    }
    for (size_t i = 2; i < dataPathVec.size() - endOffset; ++i)
    {
        auto found = dataPathVec[i].find('_');
        if (found == std::string::npos)
        {
            return "";
        }
        std::string singleId = dataPathVec[i].substr(0, found);
        if (i != 2U)
        {
            opId += "_" + singleId;
        }
        else
        {
            opId += singleId;
        }
    }
    return opId;
}

static std::string GetFullOpNameFromDataPath(const std::vector<std::string> &dataPathVec, size_t endOffset)
{
    std::string fullOpName;
    if (dataPathVec.size() <= endOffset)
    {
        return fullOpName;
    }
    for (size_t i = 2; i < dataPathVec.size() - endOffset; ++i)
    {
        auto found = dataPathVec[i].find('_');
        if (found == std::string::npos)
        {
            return "";
        }
        std::string singleOpName = dataPathVec[i].substr(found + 1);
        if (i != 2U)
        {
            fullOpName += "/" + singleOpName;
        }
        else
        {
            fullOpName += singleOpName;
        }
    }
    return fullOpName;
}

static std::string GenLineStrforStatsCsv(const BinFileInfo &fileInfo, const std::string &maxStr,
                                         const std::string &minStr, const std::string &meanStr,
                                         const std::string &normStr)
{
    std::string lineStr;
    std::vector<std::string> splitPath = SplitDataPath(fileInfo.filePath);
    if (splitPath.empty())
    {
        return lineStr;
    }

    std::string fileName = splitPath[splitPath.size() - 1];
    std::string index;
    size_t found = fileName.find('.');
    std::string inputOrOutput;
    if (fileName.find(INTENSOR_PREFIX) == 0)
    {
        inputOrOutput = "input";
        index = fileName.substr(INTENSOR_PREFIX.size(), found - INTENSOR_PREFIX.size());
    }
    else if (fileName.find(OUTTENSOR_PREFIX) == 0)
    {
        inputOrOutput = "output";
        index = fileName.substr(OUTTENSOR_PREFIX.size(), found - OUTTENSOR_PREFIX.size());
    }
    else
    {
        return lineStr;
    }
    std::string opType = fileInfo.opName;
    if (opType.rfind('/') != std::string::npos)
    {
        opType = opType.substr(opType.rfind('/') + 1);
    }

    std::string fullFilePath = GetOutDir();
    if (g_task != STATS_TASK)
    {
        fullFilePath.append(TENSOR_AND_STATS_DATA_DIR).append(fileInfo.filePath);
    }
    else
    {
        fullFilePath = "N/A";
    }

    const std::vector<std::string> colContent = {splitPath[0],    splitPath[1],  fileInfo.opName, opType,
                                                 fileInfo.opId,   inputOrOutput, index,           fileInfo.dtype,
                                                 fileInfo.format, fileInfo.dims, maxStr,          minStr,
                                                 meanStr,         normStr,       fullFilePath};

    for (const std::string &value : colContent)
    {
        bool ret = Utils::ValidateCsvString(value);
        if (!ret)
        {
            AIT_LOG_WARNING("Check input string failed! Cannot write into csv!");
            return "";
        }
        lineStr += value + ",";
    }
    lineStr.pop_back();
    AIT_LOG_DEBUG("Tensor info: " + lineStr);
    return lineStr;
}

static void ReportTensorStats(std::string &statsFilePath, const std::string &statsInfo)
{
    std::string resolvedPath = GetRealPath(statsFilePath);
    if (resolvedPath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of statsFilePath must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                      " characters, but got " + std::to_string(resolvedPath.length()));
        return;
    }
    statsFilePath = resolvedPath;

    std::ifstream f(statsFilePath, std::ios::in);
    if (!f.is_open())
    {
        std::ofstream statsFile(statsFilePath, std::ios::out);
        if (!statsFile.is_open())
        {
            AIT_LOG_WARNING("Unable to open file: " + statsFilePath);
            return;
        }

        const std::string csvHead = std::string("Device and PID,Execution Count,Op Name,Op Type,Op Id,") +
                                    "Input/Output,Index,Dtype,Format,Shape,Max,Min,Mean,Norm,Tensor Path";
        statsFile << csvHead << std::endl;
        if (!statsFile.good())
        {
            AIT_LOG_WARNING("Failed to write the ATB statistics table header");
            statsFile.close();
            return;
        }
        statsFile.close();
    }

    std::ofstream statsFile(statsFilePath, std::ios::app);
    if (!statsFile.is_open())
    {
        AIT_LOG_WARNING("Unable to open file: " + statsFilePath);
        return;
    }
    statsFile << statsInfo << std::endl;
    if (!statsFile.good())
    {
        AIT_LOG_WARNING("Failed to write the statistics of ATB operation tensor");
    }
    statsFile.close();
}

void CheckAndWriteFile(std::shared_ptr<FileSystem::BinFile> binFile, const std::string &binFilePath,
                       const std::string &statsInfo, std::string &statsFilePath)
{
    if (!statsInfo.empty())
    {
        ReportTensorStats(statsFilePath, statsInfo);
    }

    if (binFile->HasAttr("format"))
    {
        binFile->Write(binFilePath);
        AIT_LOG_DEBUG("Direct write to " + binFilePath);
        AIT_LOG_DEBUG("Saving tensor: success.");
    }
}

bool CheckOpId(const std::string &ids)
{
    for (const auto &indice : g_splitOperationIds)
    {
        if (indice.empty())
        {
            continue;
        }
        bool result = false;
        if (g_saveChild)
        {
            result = IsPrefix(ids, indice) &&
                     (ids == indice || ((ids.length() > indice.length()) && (ids[indice.length()] == '_')));
        }
        else
        {
            result = (indice == ids);
        }
        if (result)
        {
            return true;
        }
    }
    return false;
}

bool CheckOpName(const std::string &opName)
{
    std::string copyOpName = opName;
    for (char &c : copyOpName)
    {
        c = std::tolower(c);
    }

    for (const auto &indice : g_splitOperationNames)
    {
        if (indice.empty())
        {
            continue;
        }
        bool result = false;
        if (g_saveChild)
        {
            result = (IsPrefix(copyOpName, indice) || copyOpName.find("/" + indice) != std::string::npos);
        }
        else
        {
            size_t index = copyOpName.rfind('/') != std::string::npos ? copyOpName.rfind('/') + 1 : 0;
            result = (copyOpName.substr(index, indice.size()) == indice);
        }
        if (result)
        {
            return true;
        }
    }

    return false;
}

static bool IsDiskSpaceValid(const std::string path, uint64_t dataSize)
{
    unsigned long long freeSpace = 0;
    int retGetFreeSpace = GetFreeSpace(path, freeSpace);
    if (retGetFreeSpace == 1)
    {
        AIT_LOG_ERROR("Failed to get disk space for path: " + path);
        return false;
    }
    if (retGetFreeSpace == 0)
    {
        if (freeSpace <= g_minDiskSpaceFreeSize || freeSpace <= dataSize * FREE_SIZE_MULTIPLE_OF_DATA_SIZE)
        {
            AIT_LOG_ERROR(
                "Disk space is not enough, it's must more than 2G and twice size of data, free size(MB) is: " +
                std::to_string(freeSpace >> 20));
            return false;
        }
    }
    return true;
}

void SetTaskValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<std::string>(config, "task", g_task) || g_task.empty())
    {
        g_task = TENSOR_TASK;
        AIT_LOG_WARNING(std::string("\"task\" in ATB dump configuration file is empty or of wrong type, ") +
                        "which should be a string. As a result, it would be set to \"tensor\"");
    }
    if (std::find(VALID_TASKS.begin(), VALID_TASKS.end(), g_task) == VALID_TASKS.end())
    {
        AIT_LOG_WARNING(std::string("Invalid task in ATB dump configuration file, should be one of: ") +
                        "\"tensor\", \"statistics\", \"all\". As a result, it would be set to \"tensor\"");
    }
}

void SetDumpEnableValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<bool>(config, "dump_enable", g_dumpEnable))
    {
        g_dumpEnable = false;
        AIT_LOG_WARNING(std::string("\"dump_enable\" in ATB dump configuration file does not exist ") +
                        "or has a wrong type, which should be boolean. As a result, it would be set to false");
    }
}

void SetExecutionCountRangeValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<std::string>(config, "exec_range", g_executionCountRange) ||
        g_executionCountRange.empty())
    {
        g_executionCountRange = "0,0";
        AIT_LOG_WARNING(std::string("\"exec_range\" in ATB dump configuration file is empty or of wrong type, ") +
                        "which should be a string. As a result, it would be set to \"0,0\"");
    }
}

void SetOperationIdValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<std::string>(config, "ids", g_operationId))
    {
        g_operationId = "";
        AIT_LOG_WARNING(std::string("\"ids\" in ATB dump configuration file does not exist or has a wrong type, ") +
                        "which should be a string. As a result, it would be set to empty");
    }
    g_splitOperationIds = SplitString(g_operationId.c_str(), ',');
}

void SetOperationNameValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<std::string>(config, "op_name", g_operationName))
    {
        g_operationName = "";
        AIT_LOG_WARNING(std::string("\"op_name\" in ATB dump configuration file does not exist or has a wrong type, ") +
                        "which should be a string. As a result, it would be set to empty");
    }
    for (char &c : g_operationName)
    {
        c = std::tolower(c);
    }
    g_splitOperationNames = SplitString(g_operationName.c_str(), ',');
}

void SetSaveChildValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<bool>(config, "save_child", g_saveChild))
    {
        g_saveChild = false;
        AIT_LOG_WARNING(std::string("\"save_child\" in ATB dump configuration file does not exist ") +
                        "or has a wrong type, which should be boolean. As a result, it would be set to false");
    }
}

void SetDeviceValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<std::string>(config, "device", g_device))
    {
        g_device = "";
        AIT_LOG_WARNING(std::string("\"device\" in ATB dump configuration file does not exist ") +
                        "or has a wrong type, which should be a string. As a result, it would be set to empty");
    }
    g_splitDeviceIds = SplitString(g_device.c_str(), ',');
}

void SetFilterLevelValue(const nlohmann::json &config)
{
    if (!ParseJsonBaseObj2Var<int>(config, "filter_level", g_filterLevel))
    {
        g_filterLevel = FILTER_OPERATION_DATA;
        AIT_LOG_WARNING(std::string("\"filter_level\" in ATB dump configuration file does not exist ") +
                        "or has a wrong type, which should be a integer. As a result, it would be set to 1");
    }
    if (g_filterLevel < NO_FILTER_FOR_DATA || g_filterLevel > FILTER_KERNEL_DATA)
    {
        g_filterLevel = FILTER_OPERATION_DATA;
        AIT_LOG_WARNING(std::string("Invalid filter level in ATB dump configuration file, should be one of: ") +
                        "0, 1, 2. As a result, it would be set to 1");
    }
}

void SetConfigParameters(const nlohmann::json &config)
{
    SetTaskValue(config);
    SetDumpEnableValue(config);
    SetExecutionCountRangeValue(config);
    SetOperationIdValue(config);
    SetOperationNameValue(config);
    SetSaveChildValue(config);
    SetDeviceValue(config);
    SetFilterLevelValue(config);

    if (g_dumpEnable && !g_isDebugMode)
    {
        g_isDebugMode = true;
        AIT_LOG_WARNING("Ready to dump ATB data, the running speed of the model will be affected");
        std::string dataDir = GetOutDir();
        if (!dataDir.empty())
        {
            dataDir.append(TENSOR_AND_STATS_DATA_DIR);
            if (dataDir.length() > MAX_PATH_LENGTH)
            {
                AIT_LOG_ERROR("The path length of dataDir must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                              " characters, but got " + std::to_string(dataDir.length()));
                return;
            }
            bool ret = Utils::CheckDirectory(dataDir);
            if (!ret)
            {
                AIT_LOG_ERROR("Create directory failed: " + dataDir);
                return;
            }
        }
    }
}

bool atb::Probe::IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &opType)
{
    if (!g_isAtbRunning)
    {
        g_isAtbRunning = true;
        IsDiskSpaceValid(GetOutDir(), g_minDiskSpaceFreeSize / FREE_SIZE_MULTIPLE_OF_DATA_SIZE);
        const char *configPath = std::getenv("ATB_DUMP_CONFIG");
        g_configPath = configPath == nullptr ? "" : std::string(configPath);
        if (g_configPath.length() > MAX_PATH_LENGTH)
        {
            AIT_LOG_ERROR("The path length of ATB_DUMP_CONFIG must not be greater than " +
                          std::to_string(MAX_PATH_LENGTH) + " characters, but got " +
                          std::to_string(g_configPath.length()));
            g_configPath = "";
        }
        else
        {
            g_configPath = File::GetAbsPath(g_configPath);
        }
    }
    return !g_configPath.empty();
}

bool atb::Probe::IsSaveChild() { return false; }

bool atb::Probe::IsSaveTensorData() { return true; }

bool atb::Probe::IsSaveTensorDesc() { return true; }

bool atb::Probe::IsExecuteCountInRange(const uint64_t executeCount)
{
    if (!g_dumpEnable || g_executionCountRange.empty() || g_executionCountRange == "none")
    {
        return false;
    }
    if (g_executionCountRange == "all")
    {
        return true;
    }

    std::vector<std::string> saveTensorRan = SplitString(g_executionCountRange.c_str(), ',');
    for (size_t i = 1U; i < saveTensorRan.size(); i += RANGE_COUNT)
    {
        uint64_t left = static_cast<uint64_t>(SafetyStoi(saveTensorRan[i - 1].c_str(), 0));
        uint64_t right = static_cast<uint64_t>(SafetyStoi(saveTensorRan[i].c_str(), 0));
        if ((executeCount <= right) && (executeCount >= left))
        {
            return true;
        }
    }
    return false;
}

void atb::Probe::UpdateConfig()
{
    if (g_configUpdateTimes != 0 && g_configUpdateTimes != CONFIG_UPDATE_PERIOD && !g_isDebugMode)
    {
        g_configUpdateTimes++;
        return;
    }
    g_configUpdateTimes = 1;

    if (g_configPath.empty())
    {
        return;
    }
    if (!File::CheckConfigFile(g_configPath))
    {
        g_dumpEnable = false;
        return;
    }

    const double configFreshPeriod = 5;
    static auto lastReadTime = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = nowTime - lastReadTime;

    if (g_task.empty() || elapsed_seconds.count() >= configFreshPeriod)
    {
        lastReadTime = std::chrono::system_clock::now();
        nlohmann::json config;
        try
        {
            std::ifstream ifs(g_configPath, std::ios::in);
            ifs >> config;
        }
        catch (const std::exception &e)
        {
            AIT_LOG_ERROR("Json parse error! json path:" + g_configPath);
            return;
        }
        SetConfigParameters(config);
    }
}

bool atb::Probe::IsSaveTensorInSpecificDir(const std::string &tensorDir)
{
    if (!g_dumpEnable)
    {
        return false;
    }

    std::vector<std::string> splitPath = SplitDataPath(tensorDir);
    if (splitPath.empty())
    {
        return false;
    }

    if (g_operationId.empty() && g_operationName.empty())
    {
        if (!g_saveChild && splitPath.size() > LAYER_PATH_DEPTH)
        {
            return false;
        }
        return true;
    }

    if (!g_operationId.empty())
    {
        std::string currentIds = GetOpIdFromDataPath(splitPath, 0);
        if (currentIds.empty())
        {
            return false;
        }
        if (CheckOpId(currentIds))
        {
            return true;
        }
    }

    if (g_operationName.empty())
    {
        return false;
    }
    std::string fullOpName = GetFullOpNameFromDataPath(splitPath, 0);
    if (fullOpName.empty())
    {
        return false;
    }
    return CheckOpName(fullOpName);
}

bool atb::Probe::IsSaveTensorBefore() { return true; }

bool atb::Probe::IsSaveTensorAfter() { return true; }

static bool IsDeviceIdValid(const std::string &filePath)
{
    if (!g_device.empty())
    {
        size_t found = filePath.find("_");  // filePath like {device_id}_{pid}/xxx/xxx
        std::string curDeviceId = filePath.substr(0, found);
        for (const auto &indice : g_splitDeviceIds)
        {
            if (indice == curDeviceId)
            {
                return true;
            }
        }
        return false;
    }
    return true;
}

static bool IsSubString(const std::string &inputString, const std::vector<std::string> &subStrings)
{
    if (subStrings.empty())
    {
        return false;
    }
    for (const auto &subStr : subStrings)
    {
        if (inputString.find(subStr) == std::string::npos)
        {
            return false;
        }
    }
    return true;
}

static bool IsTensorFileHeadVaild(const std::string &head, const uint64_t maxHead = 50)
{
    if (head.size() > maxHead)
    {
        AIT_LOG_ERROR("The head of binfile is too long.");
        return false;
    }
    return true;
}

// helper Functions for Calculating the needed Statistics
template <typename T>
static void CalculateStatistics(const void *binData, std::pair<size_t, size_t> rangeThread, LLM::Statistics<T> &stats,
                                Mki::TensorDType tensorDType = Mki::TensorDType::TENSOR_DTYPE_UNDEFINED)
{
    size_t start = rangeThread.first;
    size_t end = rangeThread.second;
    switch (tensorDType)
    {
        case Mki::TensorDType::TENSOR_DTYPE_UNDEFINED:
        {
            const T *data = static_cast<const T *>(binData);
            for (size_t i = start; i < end; ++i)
            {
                stats.Compute(data[i]);
            }
            break;
        }
        case Mki::TensorDType::TENSOR_DTYPE_FLOAT16:
        {
            const uint16_t *data16 = static_cast<const uint16_t *>(binData);
            constexpr size_t exponentBits = 5;
            constexpr size_t mantissaBits = 10;
            for (size_t i = start; i < end; ++i)
            {
                float value = Mki::ConvertToFloat32(data16[i], exponentBits, mantissaBits);
                stats.Compute(static_cast<T>(value));
            }
            break;
        }
        case Mki::TensorDType::TENSOR_DTYPE_BF16:
        {
            const uint16_t *data16 = static_cast<const uint16_t *>(binData);
            constexpr size_t exponentBits = 8;
            constexpr size_t mantissaBits = 7;
            for (size_t i = start; i < end; ++i)
            {
                float value = Mki::ConvertToFloat32(data16[i], exponentBits, mantissaBits);
                stats.Compute(static_cast<T>(value));
            }
            break;
        }
        default:
            AIT_LOG_ERROR("Invalid datatype: " + Mki::GetDTypeStr(tensorDType));
    }
}

template <typename T>
static std::unique_ptr<LLM::StatisticsBase> GetStatisticsFromBinaryDataWithBasicType(
    const void *binData, size_t dataSize, Mki::TensorDType tensorDType = Mki::TensorDType::TENSOR_DTYPE_UNDEFINED)
{
    size_t typeSize =
        (tensorDType != Mki::TensorDType::TENSOR_DTYPE_UNDEFINED) ? Mki::GetTensorElementSize(tensorDType) : sizeof(T);
    if (typeSize == 0)
    {
        AIT_LOG_ERROR("Invalid typeSize: " + std::to_string(typeSize));
        return std::make_unique<LLM::Statistics<std::string>>();
    }
    if (dataSize == 0 || dataSize % typeSize != 0)
    {
        AIT_LOG_ERROR("Invalid dataSize: " + std::to_string(dataSize));
        return std::make_unique<LLM::Statistics<std::string>>();
    }

    size_t numElements = dataSize / typeSize;
    size_t numThreads = numElements > POOLNUM ? POOLNUM : numElements;
    size_t chunkSize = numElements / numThreads;  // Elements per thread

    // 线程池初始化
    auto &pool = GetGlobalPool(numThreads);
    std::vector<LLM::Statistics<T>> threadStats(numThreads);
    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);  // 预分配避免重分配

    // 任务分发
    for (size_t i = 0; i < numThreads; ++i)
    {
        const size_t start = i * chunkSize;
        const size_t end = (i == numThreads - 1) ? numElements : start + chunkSize;

        auto task = [i, start, end, binData, tensorDType, &threadStats]()
        { CalculateStatistics<T>(binData, std::make_pair(start, end), threadStats[i], tensorDType); };

        futures.emplace_back(pool->Enqueue(std::move(task)));
    }

    // 等待任务完成
    for (auto &future : futures)
    {
        future.get();  // 阻塞直到任务完成
    }

    auto totalStats = std::make_unique<LLM::Statistics<T>>();
    for (const auto &stats : threadStats)
    {
        (*totalStats) += stats;  // call the Overloaded operator+=
    }

    // Finalize average and L2 norm
    totalStats->ComputeAverage();
    totalStats->l2norm_ = std::sqrt(totalStats->sumOfSquares_);

    return totalStats;  // std::unique_ptr 支持派生类到基类的隐式转换
}

std::unordered_map<Mki::TensorDType, std::function<std::unique_ptr<LLM::StatisticsBase>(const void *, size_t)>>
    typeToFunctionMap = {{Mki::TensorDType::TENSOR_DTYPE_INT8, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<int8_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_INT16, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<int16_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_INT32, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<int32_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_INT64, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<int64_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_UINT8, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<uint8_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_UINT16, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<uint16_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_UINT32, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<uint32_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_UINT64, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<uint64_t>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_FLOAT, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<float>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_DOUBLE, [](const void *binData, size_t dataSize)
                          { return GetStatisticsFromBinaryDataWithBasicType<double>(binData, dataSize); }},
                         {Mki::TensorDType::TENSOR_DTYPE_FLOAT16,
                          [](const void *binData, size_t dataSize) {
                              return GetStatisticsFromBinaryDataWithBasicType<float>(
                                  binData, dataSize, Mki::TensorDType::TENSOR_DTYPE_FLOAT16);
                          }},
                         {Mki::TensorDType::TENSOR_DTYPE_BF16,
                          [](const void *binData, size_t dataSize) {
                              return GetStatisticsFromBinaryDataWithBasicType<float>(
                                  binData, dataSize, Mki::TensorDType::TENSOR_DTYPE_BF16);
                          }},
                         {Mki::TensorDType::TENSOR_DTYPE_COMPLEX64, [](const void *binData, size_t dataSize) {
                              return GetStatisticsFromBinaryDataWithBasicType<std::complex<float>>(binData, dataSize);
                          }}};

std::unique_ptr<LLM::StatisticsBase> GetStatisticsFromBinaryDataWithTensorDType(const void *binData, size_t dataSize,
                                                                                Mki::TensorDType dType)
{
    auto it = typeToFunctionMap.find(dType);
    if (Likely(it != typeToFunctionMap.end()))
    {
        return it->second(binData, dataSize);
    }
    else
    {
        AIT_LOG_WARNING("Unsupported tensor datatype: " + Mki::GetDTypeStr(dType));
        return std::make_unique<LLM::Statistics<std::string>>();
    }
}

static bool IsSaveTensorValid(const BinFileInfo &fileInfo, std::string &filePath)
{
    // 检查dataSize是否超过最大值，是否为0
    if ((fileInfo.dataSize > MsConst::MAX_FILE_SIZE_DEFAULT) || (fileInfo.dataSize == 0))
    {
        AIT_LOG_WARNING("Invalid dataSize: " + std::to_string(fileInfo.dataSize));
        return false;
    }

    filePath = GetOutDir();
    if (filePath.empty())
    {
        return false;
    }
    filePath.append(TENSOR_AND_STATS_DATA_DIR).append(fileInfo.filePath);
    if (filePath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of filePath must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                      " characters, but got " + std::to_string(filePath.length()));
        return false;
    }
    size_t found = filePath.rfind('/');
    std::string directory = filePath.substr(0, found);

    bool envValidFlag = IsDeviceIdValid(fileInfo.filePath);
    if (g_task == TENSOR_TASK || g_task == ALL_TASK)
    {
        envValidFlag =
            (envValidFlag && IsDiskSpaceValid(GetOutDir(), fileInfo.dataSize) && Utils::CheckDirectory(directory));
    }
    if (!envValidFlag)
    {
        return false;
    }

    if (!fileInfo.hostData)
    {
        AIT_LOG_WARNING("hostData is None.");
        return false;
    }

    bool isIntensorBefore = IsSubString(fileInfo.filePath, {"before", "intensor"});
    bool isOuttensorAfter = IsSubString(fileInfo.filePath, {"after", "outtensor"});
    if (!(isIntensorBefore || isOuttensorAfter))
    {
        return false;
    }

    if (!IsTensorFileHeadVaild(fileInfo.format) || !IsTensorFileHeadVaild(fileInfo.dtype) ||
        !IsTensorFileHeadVaild(fileInfo.dims))
    {
        return false;
    }

    return true;
}

static void CreateTensorBinFile(std::shared_ptr<FileSystem::BinFile> binFile, const BinFileInfo &fileInfo,
                                const std::string &opId, const std::string &opName, const std::string &fileName)
{
    binFile->AddAttr("format", fileInfo.format);
    binFile->AddAttr("dtype", fileInfo.dtype);
    binFile->AddAttr("dims", fileInfo.dims);
    if (!g_saveChild || g_filterLevel == NO_FILTER_FOR_DATA)
    {
        binFile->AddObject("data", fileInfo.hostData, fileInfo.dataSize);
        return;
    }

    const std::string layerId = opId.substr(0, opId.find('_'));
    if (layerId == opId)
    {
        g_DumpedLayerSet.insert(layerId);
    }
    if (g_DumpedLayerSet.find(layerId) == g_DumpedLayerSet.end())
    {
        binFile->AddObject("data", fileInfo.hostData, fileInfo.dataSize);
        return;
    }

    const std::string tensorKey = opId + "_" + fileName;
    if (g_realTensors.find(tensorKey) == g_realTensors.end())
    {
        if (g_filterLevel == FILTER_KERNEL_DATA && opName.substr(opName.rfind("Kernel")) == "Kernel")
        {
            binFile->AddAttr("data", fileName);
            return;
        }
        binFile->AddObject("data", fileInfo.hostData, fileInfo.dataSize);
        return;
    }

    if (g_realTensors[tensorKey] != "bin")
    {
        binFile->AddAttr("data", g_realTensors[tensorKey]);
        return;
    }

    binFile->AddObject("data", fileInfo.hostData, fileInfo.dataSize);
}

static std::string CreateTensorStats(BinFileInfo &fileInfo)
{
    const Mki::TensorFormat tensorFormat = Mki::ConvertToTensorFormat(SafetyStoi(fileInfo.format.c_str(), -1));
    const Mki::TensorDType tensorDType = Mki::ConvertToTensorDType(SafetyStoi(fileInfo.dtype.c_str(), -1));
    std::string formatStr = Mki::GetFormatStr(tensorFormat);
    std::string dtypeStr = Mki::GetDTypeStr(tensorDType);
    if (formatStr == Mki::UNDEFINED_STR)
    {
        formatStr.append("(").append(fileInfo.format).append(")");
    }
    if (dtypeStr == Mki::UNDEFINED_STR)
    {
        dtypeStr.append("(").append(fileInfo.dtype).append(")");
    }
    fileInfo.format = formatStr;
    fileInfo.dtype = dtypeStr;
    for (char &c : fileInfo.dims)
    {
        if (c == ',')
        {
            c = 'x';
        }
    }

    std::string maxStr = "N/A";
    std::string minStr = "N/A";
    std::string meanStr = "N/A";
    std::string l2normStr = "N/A";

    if (g_task == STATS_TASK || g_task == ALL_TASK)
    {
        auto stats = GetStatisticsFromBinaryDataWithTensorDType(fileInfo.hostData, fileInfo.dataSize, tensorDType);
        maxStr = stats ? stats->GetMaxStr() : "N/A";
        minStr = stats ? stats->GetMinStr() : "N/A";
        meanStr = stats ? stats->GetMeanStr() : "N/A";
        l2normStr = stats ? stats->GetL2NormStr() : "N/A";
    }
    return GenLineStrforStatsCsv(fileInfo, maxStr, minStr, meanStr, l2normStr);
}

void atb::Probe::SaveTensor(const std::string &format, const std::string &dtype, const std::string &dims,
                            const void *hostData, uint64_t dataSize, const std::string &filePath)
{
    std::vector<std::string> splitPath = SplitDataPath(filePath);
    if (splitPath.empty())
    {
        return;
    }
    std::string opId = GetOpIdFromDataPath(splitPath, 2);
    if (opId.empty())
    {
        return;
    }
    std::string opName = GetFullOpNameFromDataPath(splitPath, 2);
    if (opName.empty())
    {
        return;
    }
    std::string fileName = splitPath[splitPath.size() - 1];

    BinFileInfo fileInfo{format, dtype, dims, filePath, opId, opName, hostData, dataSize};
    std::string fullFilePath;
    if (!IsSaveTensorValid(fileInfo, fullFilePath))
    {
        return;
    }

    std::shared_ptr<FileSystem::BinFile> binFile(std::make_shared<FileSystem::BinFile>());
    if (g_task == TENSOR_TASK || g_task == ALL_TASK)
    {
        CreateTensorBinFile(binFile, fileInfo, opId, opName, fileName);
    }

    std::string statsPath = GetOutDir();
    statsPath.append(TENSOR_AND_STATS_DATA_DIR).append(splitPath.at(0)).append("/").append(splitPath.at(1));
    if (statsPath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of statsPath must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                      " characters, but got " + std::to_string(statsPath.length()));
        return;
    }
    if (!Utils::CheckDirectory(statsPath))
    {
        return;
    }

    std::string statsInfo = CreateTensorStats(fileInfo);
    statsPath.append("/").append(STATS_FILE_NAME);

    GetGlobalPool()->Enqueue(CheckAndWriteFile, binFile, fullFilePath, statsInfo, statsPath);
}

void atb::Probe::SaveTiling(const uint8_t *data, uint64_t dataSize, const std::string &filePath)
{
    if (data == nullptr)
    {
        return;
    }
    (void)dataSize;
    (void)filePath;
}

bool atb::Probe::IsSaveTiling() { return false; }

bool atb::Probe::IsSaveIntensor() { return true; }

bool atb::Probe::IsSaveOuttensor() { return true; }

bool atb::Probe::ReportOperationGraphEnable() { return true; }

static void ModifyRootNodeTensors(ordered_json &graphNodeJsonToSave, std::vector<std::string> &tensorNameList,
                                  const ordered_json &graphNodeJson)
{
    // 根节点根据自己的opName + id, 组成tensor name
    uint32_t inTensorNum = graphNodeJson["inTensorNum"].get<uint32_t>();
    uint32_t outTensorNum = graphNodeJson["outTensorNum"].get<uint32_t>();
    uint32_t internalTensorNum = (graphNodeJson.find("internalTensorNum") == graphNodeJson.end())
                                     ? 0
                                     : graphNodeJson["internalTensorNum"].get<uint32_t>();
    std::string opNameInJson = graphNodeJson["opName"].get<std::string>();

    std::string tensorName;

    for (size_t i = 0; i < inTensorNum; i++)
    {
        tensorName = opNameInJson + "_input_" + std::to_string(i);
        graphNodeJsonToSave["inTensors"].emplace_back(tensorName);
        tensorNameList.emplace_back(tensorName);
    }
    for (size_t i = 0; i < outTensorNum; i++)
    {
        tensorName = opNameInJson + "_output_" + std::to_string(i);
        graphNodeJsonToSave["outTensors"].emplace_back(tensorName);
        tensorNameList.emplace_back(tensorName);
    }
    for (size_t i = 0; i < internalTensorNum; i++)
    {
        tensorName = opNameInJson + "_internal_" + std::to_string(i);
        graphNodeJsonToSave["internalTensors"].emplace_back(tensorName);
        tensorNameList.emplace_back(tensorName);
    }

    AIT_LOG_DEBUG("tensorName: " + tensorName);

    return;
}

static bool CheckGraphInputInvalid(const std::string &opName, const ordered_json &graphNodeJson)
{
    if (graphNodeJson.find("opName") == graphNodeJson.end() || graphNodeJson.find("opType") == graphNodeJson.end() ||
        graphNodeJson.find("inTensorNum") == graphNodeJson.end() ||
        graphNodeJson.find("outTensorNum") == graphNodeJson.end())
    {
        AIT_LOG_ERROR("json parse error! opName: " + opName);
        return true;
    }

    std::string opNameInJson = graphNodeJson["opName"].get<std::string>();
    if (opNameInJson != opName)
    {
        AIT_LOG_ERROR("json parse error! opName is not equal opName in json. opName: " + opName +
                      ", opNameInJson: " + opNameInJson);
        return true;
    }
    return false;
}

void saveJsonField(const std::string &fieldName, const ordered_json &graphNodeJson, ordered_json &graphNodeJsonToSave)
{
    try
    {
        if (graphNodeJson.contains(fieldName))
        {
            graphNodeJsonToSave[fieldName] = graphNodeJson[fieldName];
        }
        else
        {
            AIT_LOG_ERROR(fieldName + " not found in graph.");
            return;
        }
    }
    catch (const std::exception &e)
    {
        AIT_LOG_ERROR("An unexpected error occurred: " + std::string(e.what()));
        return;
    }
}

void atb::Probe::ReportOperationGraph(const std::string &opName, const std::string &graph)
{
    ordered_json graphNodeJson;
    if (Utils::SafetyGuard::CheckNormalStr(opName) != SAFETY_RET::SAFE_ERR_NONE)
    {
        AIT_LOG_WARNING("Check opName string failed!");
        return;
    }
    try
    {
        graphNodeJson = ordered_json::parse(graph);
    }
    catch (const ordered_json::parse_error &ex)
    {
        AIT_LOG_WARNING("json parse error! opName:" + opName);
        AIT_LOG_WARNING("message: " + std::string(ex.what()) + '\n' + "exception id: " + std::to_string(ex.id) + '\n' +
                        "byte position of error: " + std::to_string(ex.byte));
        return;
    }

    // 检查必选项
    if (CheckGraphInputInvalid(opName, graphNodeJson))
    {
        AIT_LOG_WARNING("CheckGraphInput failed: input is invalid.");
        return;
    }

    // 保存原始json信息，用于和model拓扑合并成模型的拓扑信息
    g_layerGraphMap.SaveLayerGraph(opName, graph);

    ordered_json graphNodeJsonToSave;
    saveJsonField("opName", graphNodeJson, graphNodeJsonToSave);
    saveJsonField("opType", graphNodeJson, graphNodeJsonToSave);
    saveJsonField("param", graphNodeJson, graphNodeJsonToSave);

    // 根节点
    std::vector<std::string> tensorNameList;
    try
    {
        ModifyRootNodeTensors(graphNodeJsonToSave, tensorNameList, graphNodeJson);
    }
    catch (const std::exception &e)
    {
        AIT_LOG_WARNING("An unexpected error occurred: " + std::string(e.what()));
        return;
    }
    // 递归调用获取子节点信息
    if (graphNodeJson.find("nodes") != graphNodeJson.end())
    {
        for (auto childNodeInput : graphNodeJson["nodes"])
        {
            ordered_json childNodeToSave;
            try
            {
                DfsToModifyGraphTensors(childNodeToSave, tensorNameList, childNodeInput);
            }
            catch (const std::exception &e)
            {
                AIT_LOG_WARNING("An unexpected error occurred: " + std::string(e.what()));
                return;
            }
            graphNodeJsonToSave["nodes"].emplace_back(childNodeToSave);
        }
    }

    FilterUnnecessaryData(graphNodeJsonToSave);

    // 保存修改的Json
    std::string layerArchFilePath = GetOutDir();
    if (layerArchFilePath == "")
    {
        return;
    }

    layerArchFilePath.append("info/layer/");
    static std::string deviceId = std::to_string(GetCurrentDeviceId());
    if (deviceId != "-1")
    {
        layerArchFilePath.append(deviceId).append("_").append(std::to_string(GetCurrentProcessId()));
    }
    else
    {
        layerArchFilePath.append(std::to_string(GetCurrentProcessId()));
    }
    layerArchFilePath = GetRealPath(layerArchFilePath);
    if (!Utils::CheckDirectory(layerArchFilePath))
    {
        AIT_LOG_WARNING("Create directory failed: " + layerArchFilePath);
        return;
    }

    layerArchFilePath.append("/").append(opName).append(".json");
    if (layerArchFilePath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of layerArchFilePath must not be greater than " +
                      std::to_string(MAX_PATH_LENGTH) + " characters, but got " +
                      std::to_string(layerArchFilePath.length()));
        return;
    }
    if (File::WriteTextToFile(layerArchFilePath, graphNodeJsonToSave.dump()))
    {
        AIT_LOG_INFO("layer topo info written to file successfully! File name:" + layerArchFilePath);
    }
}

bool atb::Probe::ReportOperationStatisticEnable() { return false; }

void atb::Probe::ReportOperationSetupStatistic(const uint64_t executeCount, const std::string &opname,
                                               const std::string &st)
{
    (void)executeCount;
    (void)opname;
    (void)st;
}

void atb::Probe::ReportOperationExecuteStatistic(const uint64_t executeCount, const std::string &opname,
                                                 const std::string &st)
{
    (void)executeCount;
    (void)opname;
    (void)st;
}

static std::string MakeAbsolutePath(const std::string &path)
{
    // 如果传入的是绝对路径，则直接返回路径，如果传入的是相对路径，转化为当前程序运行所在的绝对路径
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    std::string curAbsolutePath = std::string(cwd);
    if (path.empty())
    {
        return curAbsolutePath + "/";
    }
    else if (path[0] == '/')
    {
        return path;
    }
    else if (path[0] == '~')
    {
        const char *curHomePath = std::getenv("HOME");
        std::string expandedPath = curHomePath ? (curHomePath + path.substr(1)) : path;
        return expandedPath;
    }
    else if (path == "." || path == "./")
    {
        return curAbsolutePath + "/";
    }
    else if (path.size() > 1 && path[0] == '.' && path[1] == '/')
    {
        return curAbsolutePath + path.substr(1);
    }
    return curAbsolutePath + "/" + path;
}

bool atb::Probe::ReportOperationIOTensorEnable() { return false; }

void atb::Probe::ReportOperationIOTensor(const size_t executeCount, const std::string &opName,
                                         const std::string &opParam, const std::vector<atb::Probe::Tensor> &inTensors,
                                         const std::vector<atb::Probe::Tensor> &outTensors)
{
    (void)executeCount;
    (void)opName;
    (void)opParam;
    (void)inTensors;
    (void)outTensors;
}

bool atb::Probe::ReportKernelIOTensorEnable() { return false; }

void atb::Probe::ReportKernelIOTensor(const size_t executeCount, const std::string &opName, const std::string &opParam,
                                      const std::vector<atb::Probe::Tensor> &inTensors,
                                      const std::vector<atb::Probe::Tensor> &outTensors)
{
    (void)executeCount;
    (void)opName;
    (void)opParam;
    (void)inTensors;
    (void)outTensors;
}

void atb::Probe::SaveParam(const std::string &param, const std::string &filePath)
{
    std::string fullFilePath = GetOutDir();
    if (fullFilePath == "" || !IsDeviceIdValid(filePath))
    {
        return;
    }
    fullFilePath.append(TENSOR_AND_STATS_DATA_DIR).append(filePath);
    if (fullFilePath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of fullFilePath must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                      " characters, but got " + std::to_string(fullFilePath.length()));
        return;
    }
    size_t found = fullFilePath.rfind('/');
    std::string fileName = fullFilePath.substr(found);
    fullFilePath = GetRealPath(fullFilePath.substr(0, found));
    if (fullFilePath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of fullFilePath must not be greater than " + std::to_string(MAX_PATH_LENGTH) +
                      " characters, but got " + std::to_string(fullFilePath.length()));
        return;
    }
    if (!Utils::CheckDirectory(fullFilePath))
    {
        AIT_LOG_WARNING("Create directory failed: " + fullFilePath);
        return;
    }

    File::WriteTextToFile(fullFilePath.append(fileName), param);
}

bool atb::Probe::IsSaveParam() { return true; }

/****************************************************************************************\
                                    算子溢出检测 AIT 接口
\****************************************************************************************/

bool atb::Probe::IsOverflowCheck() { return false; }

bool atb::Probe::IsOverflowStop() { return false; }

void atb::Probe::ReportOverflowKernel(const std::string &kernelPath) { (void)kernelPath; }
}  // end of namespace atb

namespace atb_speed
{
struct ModelGraphMap
{
    std::map<std::string, std::string> modelGraphMap_;

    bool IsInitModelGraph(const std::string &modelName)
    {
        auto it = modelGraphMap_.find(modelName);
        return (it == modelGraphMap_.end()) ? true : false;
    };

    void SaveModelGraph(const std::string &modelName, const std::string &graph) { modelGraphMap_[modelName] = graph; };
};
ModelGraphMap g_modelGraphMap;

bool atb_speed::SpeedProbe::IsReportModelTopoInfo(const std::string &modelName)
{
    // 只保存一次
    if (Utils::SafetyGuard::CheckNormalStr(modelName) != SAFETY_RET::SAFE_ERR_NONE)
    {
        AIT_LOG_ERROR("Check modelName string failed!");
        return false;
    }
    return g_modelGraphMap.IsInitModelGraph(modelName);
}

void atb_speed::SpeedProbe::ReportModelTopoInfo(const std::string &modelName, const std::string &graph)
{
    ordered_json modelJson;
    if (Utils::SafetyGuard::CheckNormalStr(modelName) != SAFETY_RET::SAFE_ERR_NONE)
    {
        AIT_LOG_WARNING("Check modelName string failed!");
        return;
    }
    g_modelGraphMap.SaveModelGraph(modelName, graph);

    try
    {
        modelJson = ordered_json::parse(graph);
    }
    catch (ordered_json::parse_error &ex)
    {
        AIT_LOG_WARNING("parse model topo info error! modelName: " + modelName);
        AIT_LOG_WARNING("message: " + std::string(ex.what()) + "\nexception id: " + std::to_string(ex.id) +
                        "\nbyte position of error: " + std::to_string(ex.byte));
        return;
    }

    // 和atb保存的layer拓扑信息进行合并
    if (modelJson.find("nodes") != modelJson.end())
    {
        for (auto &layerJson : modelJson["nodes"])
        {
            try
            {
                MergeLayerTopoInfo(layerJson);
            }
            catch (const std::exception &e)
            {
                AIT_LOG_WARNING("An unexpected error occurred: " + std::string(e.what()));
                return;
            }
        }
    }

    // 保存合并后的Json
    std::string modelArchFilePath = GetOutDir();
    if (modelArchFilePath == "")
    {
        return;
    }

    modelArchFilePath.append("info/model/");
    static std::string deviceId = std::to_string(GetCurrentDeviceId());
    if (deviceId != "-1")
    {
        modelArchFilePath.append(deviceId).append("_").append(std::to_string(GetCurrentProcessId()));
    }
    else
    {
        modelArchFilePath.append(std::to_string(GetCurrentProcessId()));
    }
    modelArchFilePath = GetRealPath(modelArchFilePath);
    if (!Utils::CheckDirectory(modelArchFilePath))
    {
        AIT_LOG_WARNING("Create directory failed: " + modelArchFilePath);
        return;
    }

    modelArchFilePath.append("/").append(modelName).append(".json");
    if (modelArchFilePath.length() > MAX_PATH_LENGTH)
    {
        AIT_LOG_ERROR("The path length of modelArchFilePath must not be greater than " +
                      std::to_string(MAX_PATH_LENGTH) + " characters, but got " +
                      std::to_string(modelArchFilePath.length()));
        return;
    }
    if (File::WriteTextToFile(modelArchFilePath, modelJson.dump()))
    {
        AIT_LOG_INFO("model topo info written to file successfully! File name: " + modelArchFilePath);
    }
}
}  // end of namespace atb_speed
