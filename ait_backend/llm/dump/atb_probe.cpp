/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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

#include "atb_probe.h"
#include <syscall.h>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
// for calculating dumped-tensor statistics
#include <vector>
#include <thread>
#include <algorithm>
#include <functional>
#include <memory>
#include <complex>
#include <utility>
#include <chrono>
#include "Statistics.h"
// endfor
#include <unistd.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include "bin_file.h"
#include "nlohmann/json.hpp"
#include "ait_logger.h"
#include "utils.h"
#include "umask_wrapper.h"
#include "DumpThreadPool.h"
#include "const.h"
#include "safety_guard.h"
#include "file_registry.h"
#include "file.h"

using ordered_json = nlohmann::ordered_json;
using MsConst::SAFETY_RET;

// C++11标准不支持make_unique函数，此处为make_unique的简单实现
#if __cplusplus < 201402L
namespace std {
    // 非数组类型
    template<typename T, typename... Args>
    typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    // 数组类型(T[])
    template <typename T>
    using EnableIfDynamicArray = typename std::enable_if<
        std::is_array<T>::value && std::extent<T>::value == 0,
        std::unique_ptr<T>
    >::type;

    template<typename T>
    EnableIfDynamicArray<T> make_unique(size_t size)
    {
        using U = typename std::remove_extent<T>::type;
        return std::unique_ptr<T>(new U[size]);
    }

    // 禁用定长数组(如T[5])
    template<typename T, typename... Args>
    typename std::enable_if<std::extent<T>::value != 0, std::unique_ptr<T>>::type make_unique(Args&&...) = delete;
}
#endif

namespace {
    unsigned long long g_minDiskSpaceFreeSize = 2147483648; // 2G
    constexpr size_t FREE_SIZE_MULTIPLE_OF_DATA_SIZE = 2; // free size至少两倍data size大小
    static const int POOLNUM = 12;
    constexpr uint32_t g_maxDeep = 1000;

    template <typename T>
    inline bool Likely(T&& condition) noexcept
    {
        return __builtin_expect(static_cast<bool>(condition), 1);
    }

    template <typename T>
    inline bool Unlikely(T&& condition) noexcept
    {
        return __builtin_expect(static_cast<bool>(condition), 0);
    }

    static int SafetyStoi(const char* env, int defalutValue)
    {
        int ans = defalutValue;
        try {
            ans = std::stoi(env);
        } catch (const std::invalid_argument& e) {
            AIT_LOG_ERROR("Cannot converting environment varible to int.");
            return ans;
        } catch (const std::out_of_range& e) {
            AIT_LOG_ERROR("The value of environment is illegal.");
            return ans;
        }
        return ans;
    }

    struct LayerGraphMap {
        std::map<std::string, std::string> layerGraphMap_;

        void SaveLayerGraph(const std::string &opName, const std::string &graph)
        {
            layerGraphMap_[opName] = graph;
        };

        std::string GetLayerGraph(const std::string &opName)
        {
            auto it = layerGraphMap_.find(opName);
            return (it == layerGraphMap_.end()) ? "" : it->second;
        };
    };

    std::unique_ptr<FileRegistry::FileRegistry>& GetFileRegistry()
    {
        static std::unique_ptr<FileRegistry::FileRegistry> instance =
            std::make_unique<FileRegistry::FileRegistry>();
        return instance;
    }

    std::unique_ptr<ThreadPool::DumpThreadPool>& GetGlobalPool(size_t numThreads = POOLNUM)
    {
        static std::unique_ptr<ThreadPool::DumpThreadPool> instance =
            std::make_unique<ThreadPool::DumpThreadPool>(numThreads);
        return instance;
    }
}

static int GetFreeSpace(std::string path, unsigned long long *freeSpace)
{
    struct statvfs diskInfo;

    if (statvfs(path.c_str(), &diskInfo) == -1) {
        AIT_LOG_ERROR("statvfs() error:" + Utils::GetLastErrorStr());
        return 1;
    }
    *freeSpace = diskInfo.f_bavail * diskInfo.f_bsize;
    return 0;
}

static int32_t GetCurrentProcessId()
{
    int32_t pid = getpid();
    if (pid == -1) {
        AIT_LOG_ERROR("get pid failed");
    }
    return pid;
}

static bool IsPrefix(const std::string &str, const std::string &prefix)
{
    return str.compare(0, prefix.length(), prefix) == 0;
}

static const std::string& GetOutDir()
{
    static std::string outDir = "";
    if (outDir == "") {
        const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
        outDir = (outputDir != nullptr ? outputDir : "./");
        outDir = GetRealPath(outDir);
        const char* timestamp = std::getenv("ATB_TIMESTAMP");
        const std::string timestampStr = (timestamp != nullptr ? timestamp : "");
        if (timestampStr == "") {
            outDir = outDir + "msit_dump/";
        } else {
            outDir = outDir + "msit_dump_" + timestampStr + "/";
        }
        bool ret = Utils::CheckDirectory(outDir);
        if (!ret) {
            AIT_LOG_ERROR("Create directory failed: " + outDir);
            outDir = "";
        }
    }
    return outDir;
}

static bool IsInTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind("/");
    std::string fileName = filePath;
    if (sepPos != std::string::npos) {
        fileName.erase(0, sepPos + 1U);
    }
    bool flag = (fileName.find("intensor") != std::string::npos) || (fileName.find("inTensor") != std::string::npos);
    AIT_LOG_DEBUG("IsInTensorBinPath: " + std::to_string(flag));
    return flag;
}

static bool IsOutTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind("/");
    std::string fileName = filePath;
    if (sepPos != std::string::npos) {
        fileName.erase(0, sepPos + 1U);
    }
    bool flag = (fileName.find("outtensor") != std::string::npos) || (fileName.find("outTensor") != std::string::npos);
    AIT_LOG_DEBUG("IsOutTensorBinPath: " + std::to_string(flag));
    return flag;
}

static bool IsSaveDumpType(const std::string &tar)
{
    const char* dumpTypeList = std::getenv("ATB_DUMP_TYPE");

    if (dumpTypeList != nullptr) {
        AIT_LOG_DEBUG("Got ATB_DUMP_TYPE: " + std::string(dumpTypeList));

        std::vector<std::string> dumpTypes = SplitString(dumpTypeList, '|');
        for (const auto &type : dumpTypes) {
            if (type == tar) {
                AIT_LOG_DEBUG(tar + " is found.");
                return true;
            }
        }
    }

    return false;
}

static void DfsToModifyGraphTensors(ordered_json &curNodeToSave,
    const std::vector<std::string> &fatherNodeTensorNameList, const ordered_json &curNodeInput, uint32_t curDeep = 1)
{
    if (curDeep >= g_maxDeep) {
        AIT_LOG_ERROR("The depth of the current graph structure is too deep, and the traversal node fails");
        throw std::runtime_error("The maximum recursive depth exceeds " + std::to_string(g_maxDeep) + ".");
    }
    std::string opName = curNodeInput["opName"].get<std::string>();
    curNodeToSave["opName"] = curNodeInput["opName"];
    curNodeToSave["opType"] = curNodeInput["opType"];
    curNodeToSave["param"] = curNodeInput["param"];

    std::vector<std::string> curNodeTensorNameList;

    // 子节点的inTensors\outTensors为父节点的fatherNodeTensorNameList子集, 根据inTensorIds、outTensorIds获取
    for (auto item : curNodeInput["inTensorIds"]) {
        uint32_t inputIndex = item.get<uint32_t>();
        if (inputIndex >= fatherNodeTensorNameList.size()) {
            AIT_LOG_ERROR("inputIndex out of fatherNodeTensorNameList: " + opName);
            return;
        }
        curNodeToSave["inTensors"].emplace_back(fatherNodeTensorNameList[inputIndex]);
        curNodeTensorNameList.emplace_back(fatherNodeTensorNameList[inputIndex]);
    }

    for (auto item : curNodeInput["outTensorIds"]) {
        uint32_t outputIndex = item.get<uint32_t>();
        if (outputIndex >= fatherNodeTensorNameList.size()) {
            AIT_LOG_ERROR("outputIndex out of fatherNodeTensorNameList: " + opName);
            return;
        }
        curNodeToSave["outTensors"].emplace_back(fatherNodeTensorNameList[outputIndex]);
        curNodeTensorNameList.emplace_back(fatherNodeTensorNameList[outputIndex]);
    }

    // 子节点的internalTensors根据自己的opName + id, 组成tensor name
    uint32_t internalTensorNum = (curNodeInput.find("internalTensorNum") == curNodeInput.end()) ?
                                  0 : curNodeInput["internalTensorNum"].get<uint32_t>();
    for (size_t i = 0; i < internalTensorNum; i++) {
        std::string tensorName = opName + "_internal_" + std::to_string(i);
        curNodeToSave["internalTensors"].emplace_back(tensorName);
        curNodeTensorNameList.emplace_back(tensorName);
    }

    // 递归调用获取子节点信息
    if (curNodeInput.find("nodes") != curNodeInput.end()) {
        for (auto childNodeInput : curNodeInput["nodes"]) {
            ordered_json childNodeToSave;
            DfsToModifyGraphTensors(childNodeToSave, curNodeTensorNameList, childNodeInput, curDeep + 1);
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
    std::string opName = layerJson["opName"].get<std::string>()+ "_" + std::to_string(g_aitOperationBaseId++);
    layerJson["opName"] = opName;
    std::string atbLayerGraph = g_layerGraphMap.GetLayerGraph(opName);
    if (atbLayerGraph == "") {
        return;
    }
    ordered_json atbLayerJson;
    // inTensor和outTensor从model里获取，internalTensor自己申请id, 组成layerTensorNameList
    try {
        atbLayerJson = ordered_json::parse(atbLayerGraph);
    } catch (const ordered_json::parse_error& ex) {
        AIT_LOG_ERROR("json parse error! opName:" + opName);
        AIT_LOG_ERROR("message: " + std::string(ex.what()) + '\n' + "exception id: " + std::to_string(ex.id) + '\n' +
               "byte position of error: " + std::to_string(ex.byte));
        return;
    }
    std::vector<std::string> layerTensorNameList;
    for (auto item : layerJson["inTensors"]) {
        layerTensorNameList.emplace_back(item);
    }

    for (auto item : layerJson["outTensors"]) {
        layerTensorNameList.emplace_back(item);
    }

    uint32_t internalTensorNum = (atbLayerJson.find("internalTensorNum") == atbLayerJson.end()) ?
                                  0 : atbLayerJson["internalTensorNum"].get<uint32_t>();
    for (size_t i = 0; i < internalTensorNum; i++) {
        std::string tensorName = opName + "_internal_" + std::to_string(i);
        layerJson["internalTensors"].emplace_back(tensorName);
        layerTensorNameList.emplace_back(tensorName);
    }

    // 递归调用获取每个layer的子节点信息
    if (atbLayerJson.find("nodes") != atbLayerJson.end()) {
        for (auto childNodeInput : atbLayerJson["nodes"]) {
            ordered_json childNodeToSave;
            DfsToModifyGraphTensors(childNodeToSave, layerTensorNameList, childNodeInput);
            layerJson["nodes"].emplace_back(childNodeToSave);
        }
    }
    return;
}

void SaveSubProcessInfo(const std::string infoToSave)
{
    // 将子进程的信息保存给ait侧
    const char *outputDir = std::getenv("ATB_DUMP_SUB_PROC_INFO_SAVE_PATH");
    if (outputDir == nullptr) {
        return;
    }

    std::string outDir = outputDir;
    outDir = GetRealPath(outDir);
    bool ret = Utils::CheckDirectory(outDir);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + outDir);
        return;
    }

    std::string outPath = outDir + "/subprocess_info.txt";
    ms::UmaskWrapper uw;
    std::ofstream outfile(outPath, std::ios::app);

    if (outfile.is_open()) {
        outfile << infoToSave << std::endl;
        outfile.close();
    }
    return;
}

namespace FileRegistry {

    bool FileRegistry::enableSymlink_ = false;
    
    const std::string* FileRegistry::RegisterFile(const uint64_t hash,
                                                  const std::string& path)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = registry_.find(hash);
        if (it != registry_.end()) {
            return &(it->second);
        }
        registry_[hash] = path;
        return nullptr;
    }

    void FileRegistry::UpdateEnableSymlink()
    {
        enableSymlink_ = [] {
            const char* env = std::getenv("ATB_SAVE_SYMLINK"); // 软链接环境变量开关
            if (env != nullptr) {
                int value = SafetyStoi(env, 0);
                if (value == atb::SAVE_SYMLINK) {
                    return true;
                }
            }
            return false;
        }();
    }
}

namespace Mki {

    size_t GetTensorElementSize(const TensorDType dtype)
    {
        auto iter = MAP_OF_DTYPE_SIZE.find(dtype);
        if (iter == MAP_OF_DTYPE_SIZE.end()) {
            AIT_LOG_ERROR("Get Tensor ElementSize:dtype not found!");
            return 0;
        }
        return iter->second;
    }

    TensorDType GetDTypeWithStr(const std::string &typeStr)
    {
        auto it = MAP_STRING_TO_DTYPE.find(typeStr);
        if (it != MAP_STRING_TO_DTYPE.end()) {
            return it->second;
        }
        return TensorDType::TENSOR_DTYPE_UNDEFINED;
    }

    const std::string &GetStrWithDType(int dType)
    {
        auto it = MAP_DTYPE_TO_STRING.find(dType);
        if (it != MAP_DTYPE_TO_STRING.end()) {
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
        if (rawExponent == (1 << exponentBits) - 1) { // All 1s represent NaN or infinity
            if (mantissa != 0) {
                return std::numeric_limits<float>::quiet_NaN(); // NaN
            } else {
                return sign * std::numeric_limits<float>::infinity(); // Infinity
            }
        } else if (rawExponent == 0) { // Exponents with all zeros indicate non-normalized numbers or zeros
            if (mantissa == 0) {
                return sign * 0.0f; // Zero
            } else {
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
}

namespace atb {

static std::string g_configPath;
static std::string g_dumpEnable;
static std::string g_tensorIdsStr;
static std::vector<std::string> g_splitTensorIds;
static bool g_isSaveChild;
static std::string g_tensorRangeStr;
static std::string g_dumpDevice;
static std::vector<std::string> g_splitDeviceIds;

void CheckAndWriteFile(std::shared_ptr<FileSystem::BinFile> binFile,
                       const std::string &outputPath)
{
    auto& fileRegistry = GetFileRegistry();
    // 优先处理软链接关闭的情况
    if (!fileRegistry->GetEnableSymlink()) {
        binFile->Write(outputPath);
        AIT_LOG_DEBUG("Direct write to " + outputPath);
        AIT_LOG_DEBUG("Saving tensor: success.");
        return;
    }
    
    ms::UmaskWrapper umask_guard;
    const uint64_t fileHash = binFile->CalcHash();
    // 原子化注册文件
    if (const std::string* existingPath = fileRegistry->RegisterFile(fileHash, outputPath)) {
        // 创建符号链接
        if (::symlink(existingPath->c_str(), outputPath.c_str()) != 0) {
            AIT_LOG_ERROR("Error creating symlink from " + (*existingPath) +
                " to " + outputPath + ": " + Utils::GetLastErrorStr());
            return;
        }
        AIT_LOG_DEBUG("Create tensor symlink success: " + outputPath + " -> " + *existingPath);
    } else {
        binFile->Write(outputPath);
        AIT_LOG_DEBUG("Direct write to " + outputPath);
    }
    AIT_LOG_DEBUG("Saving tensor: success.");
}

bool CheckOpType(const std::string &opType, bool withId, const std::string &typeStr)
{
    std::string copyOpType = opType;
    for (char &c : copyOpType) {
        c = std::tolower(c);
    }

    if (withId) {
        size_t found = copyOpType.find_first_of("_");
        if (found == std::string::npos) {
            return false;
        }
        copyOpType = copyOpType.substr(found + 1);
    }

    if (!typeStr.empty()) {
        std::vector<std::string> splitTid = SplitString(typeStr.c_str(), ',');
        for (const auto &indice : splitTid) {
            if (IsPrefix(copyOpType, indice)) {
                return true;
            }
        }
    }
    return false;
}

bool atb::Probe::IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &opType)
{
    if (!IsSaveDumpType("tensor")) {return false;}
    if (!g_configPath.empty()) {return true;}

    std::string vidStr;
    const char *vid = std::getenv("ATB_SAVE_TENSOR_IDS"); // 应该是20_1_9,1_23,5_29_1
    if (vid != nullptr) {
        vidStr = std::string(vid);
    }
    std::string tidStr;
    const char *tid = std::getenv("ATB_SAVE_TENSOR_RUNNER"); // 应该是LinearOps，SelfAttention
    if (tid != nullptr) {
        tidStr = std::string(tid);
    }
    if (vidStr.empty() && tidStr.empty()) {
        return true;
    }

    if (!vidStr.empty()) {
        std::vector<std::string> splitVid = SplitString(vidStr.c_str(), ',');
        std::string query = "";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i != 0U) {
                query += "_" + std::to_string(ids[i]);
            } else {
                query += std::to_string(ids[i]);
            }
        }
        for (const auto &indice : splitVid) {
            bool result = false;
            if (IsSaveChild()) {
                result = IsPrefix(query, indice) &&
                         (query == indice ||
                         ((query.length() > indice.length()) &&
                         (query[indice.length()] == '_')));
            } else {
                result = (indice == query);
            }
            if (result) {
                return true;
            }
        }
    }

    return CheckOpType(opType, false, tidStr);
}

bool atb::Probe::IsSaveChild()
{
    const char* child = std::getenv("ATB_SAVE_CHILD");
    if (child == nullptr) {
        return false;
    }
    return SafetyStoi(child, 0) != 0;
}

bool atb::Probe::IsSaveTensorData()
{
    const char* saveTensor = std::getenv("ATB_SAVE_TENSOR");
    if (saveTensor != nullptr) {
        int value = SafetyStoi(saveTensor, 0);
        if (value == SAVE_TENSOR_DATA) {
            return true;
        }
    }
    return false;
}

bool atb::Probe::IsSaveTensorDesc()
{
    return true;
}

bool atb::Probe::IsExecuteCountInRange(const uint64_t executeCount)
{
    const char *saveTensorRange = std::getenv("ATB_SAVE_TENSOR_RANGE");
    if (g_configPath.empty()) {
        g_tensorRangeStr = saveTensorRange == nullptr ? "" : std::string(saveTensorRange);
    }
    if (g_tensorRangeStr.empty()) {return false;}
    if (g_tensorRangeStr == "all") { return true;}

    std::vector<std::string> saveTensorRan = SplitString(g_tensorRangeStr.c_str(), ',');
    for (size_t i = 1U; i < saveTensorRan.size(); i += RANGE_COUNT) {
        uint64_t left = static_cast<uint64_t>(SafetyStoi(saveTensorRan[i - 1].c_str(), 0));
        uint64_t right = static_cast<uint64_t>(SafetyStoi(saveTensorRan[i].c_str(), 0));
        if ((executeCount <= right) && (executeCount >= left)) {
            return true;
        }
    }
    return false;
}

void atb::Probe::UpdateConfig()
{
    const char *configPath = std::getenv("ATB_DUMP_CONFIG");
    g_configPath = (configPath != nullptr ? File::GetFullPath(std::string(configPath)) : "");
    if (g_configPath.empty()) {return;}

    const double configFreshPeriod = 5;
    static auto lastReadTime = std::chrono::system_clock::now();

    auto nowTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = nowTime - lastReadTime;

    if (g_dumpEnable.empty() || elapsed_seconds.count() >= configFreshPeriod) {
        lastReadTime = std::chrono::system_clock::now();
        if (!File::IsPathExist(g_configPath) || !File::IsFileReadable(g_configPath)) {
            AIT_LOG_ERROR("Valid ATB dump config: " + g_configPath);
            return;
        }
        nlohmann::json config;
        try {
            std::ifstream ifs(g_configPath, std::ios::in);
            ifs >> config;
        } catch (...) {
            AIT_LOG_ERROR("Json parse error! json path:" + g_configPath);
            return;
        }

        nlohmann::json::const_iterator iter = config.find("dump_enable");
        g_dumpEnable = iter != config.end() && !iter->get<std::string>().empty() ? iter->get<std::string>(): "false";

        iter = config.find("ids");
        g_tensorIdsStr = iter != config.end() ? iter->get<std::string>() : "";
        g_splitTensorIds = SplitString(g_tensorIdsStr.c_str(), ',');

        iter = config.find("child");
        if (iter == config.end() || iter->get<std::string>().empty()) {
            g_isSaveChild = true;
        } else {
            g_isSaveChild = (iter->get<std::string>() == "true");
        }

        iter = config.find("er");
        if (iter == config.end() || iter->get<std::string>().empty()) {
            g_tensorRangeStr = "0,0";
        } else {
            g_tensorRangeStr = iter->get<std::string>();
        }

        iter = config.find("device");
        g_dumpDevice = iter != config.end() ? iter->get<std::string>() : "";
        g_splitDeviceIds = SplitString(g_dumpDevice.c_str(), ',');
    }
}

bool atb::Probe::IsSaveTensorInSpecificDir(const std::string &tensorDir)
{
    if (g_configPath.empty()) {return true;}
    if (!g_dumpEnable.empty() && g_dumpEnable != "true") {return false;}

    std::vector<std::string> splitPath = SplitString(tensorDir.c_str(), '/');
    constexpr size_t minSizeOfDirSegs = 3;
    if (splitPath.size() < minSizeOfDirSegs) {
        AIT_LOG_WARNING("Expected a valid parent directory of tensors, but got \"" + tensorDir + "\"");
        return false;
    }

    std::string operationType;
    const char *tid = std::getenv("ATB_SAVE_TENSOR_RUNNER"); // 应该是LinearOps，SelfAttention
    if (tid != nullptr) {
        operationType = std::string(tid);
    }
    if (g_tensorIdsStr.empty() && operationType.empty()) {
        if (!g_isSaveChild && splitPath.size() > minSizeOfDirSegs) {return false;}
        return true;
    }

    if (!g_tensorIdsStr.empty()) {
        std::string currentIds;
        for (size_t i = 2; i < splitPath.size(); ++i) {
            size_t found = splitPath[i].find_first_of("_");
            if (found == std::string::npos) {
                return false;
            }
            std::string singleId = splitPath[i].substr(0, found);
            if (i != 2U) {
                currentIds += "_" + singleId;
            } else {
                currentIds += singleId;
            }
        }
        for (const auto &indice : g_splitTensorIds) {
            bool result = false;
            if (g_isSaveChild) {
                result = IsPrefix(currentIds, indice) &&
                         (currentIds == indice ||
                         ((currentIds.length() > indice.length()) &&
                         (currentIds[indice.length()] == '_')));
            } else {
                result = (indice == currentIds);
            }
            if (result) {
                return true;
            }
        }
    }

    return CheckOpType(splitPath[splitPath.size() - 1], true, operationType);
}

bool atb::Probe::IsSaveTensorBefore()
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    int value = (saveTensorTime) ? SafetyStoi(saveTensorTime, 0) : SAVE_TENSOR_AFTER;
    const char* saveTensorInBeforeOutAfter = std::getenv("ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER");

    int saveFlag = (saveTensorInBeforeOutAfter) ? SafetyStoi(saveTensorInBeforeOutAfter, 0) :
                    SAVE_TENSOR_IN_BEFORE_OUT_AFTER;

    bool isSaveBefore = (value == SAVE_TENSOR_BEFORE || value == SAVE_TENSOR_BOTH ||
                        saveFlag == SAVE_TENSOR_IN_BEFORE_OUT_AFTER);

    AIT_LOG_DEBUG("IsSaveTensorBefore: " + std::to_string(isSaveBefore));
    return isSaveBefore;
}

bool atb::Probe::IsSaveTensorAfter()
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    int value = (saveTensorTime) ? SafetyStoi(saveTensorTime, 0) : SAVE_TENSOR_AFTER;
    const char* saveTensorInBeforeOutAfter = std::getenv("ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER");

    int saveFlag = (saveTensorInBeforeOutAfter) ? SafetyStoi(saveTensorInBeforeOutAfter, 0) :
                    SAVE_TENSOR_IN_BEFORE_OUT_AFTER;

    bool isSaveAfter = (value == SAVE_TENSOR_AFTER || value == SAVE_TENSOR_BOTH ||
                        saveFlag == SAVE_TENSOR_IN_BEFORE_OUT_AFTER);

    AIT_LOG_DEBUG("IsSaveTensorAfter: " + std::to_string(isSaveAfter));
    return isSaveAfter;
}

static bool IsDeviceIdValid(const std::string &filePath)
{
    if (g_configPath.empty()) {
        const char* saveDeviceId = std::getenv("ATB_DEVICE_ID");
        g_dumpDevice = saveDeviceId != nullptr ? std::string(saveDeviceId) : "";
        g_splitDeviceIds = SplitString(g_dumpDevice.c_str(), ',');
    }
    if (!g_dumpDevice.empty()) {
        size_t found = filePath.find("_");  // filePath like {device_id}_{pid}/xxx/xxx
        std::string curDeviceId = filePath.substr(0, found);
        for (const auto &indice : g_splitDeviceIds) {
            if (indice == curDeviceId) {
                return true;
            }
        }
        return false;
    }
    return true;
}

static bool IsDiskSpaceValid(const std::string path, uint64_t dataSize)
{
    unsigned long long freeSpace = 0;
    int retGetFreeSpace = GetFreeSpace(path, &freeSpace);
    if (retGetFreeSpace == 1) {
        AIT_LOG_ERROR("Failed to get disk space for path: " + path);
        return false;
    }
    if (retGetFreeSpace == 0) {
        if (freeSpace <= g_minDiskSpaceFreeSize || freeSpace <= dataSize * FREE_SIZE_MULTIPLE_OF_DATA_SIZE) {
            AIT_LOG_ERROR("Disk space is not enough, it's must more than 2G, free size(MB) is: " +
            std::to_string(freeSpace >> 20));
            return false;
        }
    }
    return true;
}

static bool IsSubString(const std::string& inputString, const std::vector<std::string>& subStrings)
{
    if (subStrings.empty()) {
        return false;
    }
    for (const auto& subStr : subStrings) {
        if (inputString.find(subStr) == std::string::npos) {
            return false;
        }
    }
    return true;
}

static bool IsTensorFileHeadVaild(const std::string &head, const uint64_t maxHead = 50)
{
    if (head.size() > maxHead) {
        AIT_LOG_ERROR("The head of binfile is too long.");
        return false;
    }
    return true;
}

// helper Functions for Calculating the needed Statistics
struct BinFileInfo {
    const std::string format;
    const std::string dtype;
    const std::string dims;
    const void *hostData;
    const uint64_t dataSize;
    const std::string filePath;
};

template<typename T>
static void CalculateStatistics(const void* binData,
    std::pair<size_t, size_t> rangeThread, LLM::Statistics<T> &stats,
    Mki::TensorDType tensorDType = Mki::TensorDType::TENSOR_DTYPE_UNDEFINED)
{
    size_t start = rangeThread.first;
    size_t end = rangeThread.second;
    switch (tensorDType) {
        case Mki::TensorDType::TENSOR_DTYPE_UNDEFINED: {
            const T* data = static_cast<const T*>(binData);
            for (size_t i = start; i < end; ++i) {
                stats.Compute(data[i]);
            }
            break;
        }
        case Mki::TensorDType::TENSOR_DTYPE_FLOAT16: {
            const uint16_t* data16 = static_cast<const uint16_t*>(binData);
            constexpr size_t exponentBits = 5;
            constexpr size_t mantissaBits = 10;
            for (size_t i = start; i < end; ++i) {
                float value = Mki::ConvertToFloat32(data16[i], exponentBits, mantissaBits);
                stats.Compute(static_cast<T>(value));
            }
            break;
        }
        case Mki::TensorDType::TENSOR_DTYPE_BF16: {
            const uint16_t* data16 = static_cast<const uint16_t*>(binData);
            constexpr size_t exponentBits = 8;
            constexpr size_t mantissaBits = 7;
            for (size_t i = start; i < end; ++i) {
                float value = Mki::ConvertToFloat32(data16[i], exponentBits, mantissaBits);
                stats.Compute(static_cast<T>(value));
            }
            break;
        }
        default:
            AIT_LOG_ERROR("Invalid datatype: " + Mki::GetStrWithDType(tensorDType));
    }
}

template<typename T>
static std::unique_ptr<LLM::StatisticsBase> GetStatisticsFromBinaryDataWithBasicType(
    const void *binData, size_t dataSize,
    Mki::TensorDType tensorDType = Mki::TensorDType::TENSOR_DTYPE_UNDEFINED)
{
    size_t typeSize = (tensorDType != Mki::TensorDType::TENSOR_DTYPE_UNDEFINED) ?
                    Mki::GetTensorElementSize(tensorDType) : sizeof(T);
    if (typeSize == 0) {
        AIT_LOG_ERROR("Invalid typeSize: " + std::to_string(typeSize));
        return std::make_unique<LLM::Statistics<std::string>>();
    }
    if (dataSize == 0 || dataSize % typeSize != 0) {
        AIT_LOG_ERROR("Invalid dataSize: " + std::to_string(dataSize));
        return std::make_unique<LLM::Statistics<std::string>>();
    }

    size_t numElements = dataSize / typeSize;
    size_t numThreads = numElements > POOLNUM ? POOLNUM : numElements;
    size_t chunkSize = numElements / numThreads; // Elements per thread
    
    // 线程池初始化
    auto& pool = GetGlobalPool(numThreads);
    std::vector<LLM::Statistics<T>> threadStats(numThreads);
    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);  // 预分配避免重分配

    // 任务分发
    for (size_t i = 0; i < numThreads; ++i) {
        const size_t start = i * chunkSize;
        const size_t end = (i == numThreads - 1) ? numElements : start + chunkSize;

        auto task = [i, start, end, binData, tensorDType, &threadStats]() {
            CalculateStatistics<T>(binData, std::make_pair(start, end), threadStats[i], tensorDType);
        };

        futures.emplace_back(pool->Enqueue(std::move(task)));
    }

    // 等待任务完成
    for (auto& future : futures) {
        future.get();  // 阻塞直到任务完成
    }

    auto totalStats = std::make_unique<LLM::Statistics<T>>();
    for (const auto& stats : threadStats) {
        (*totalStats) += stats; // call the Overloaded operator+=
    }

    // Finalize average and L2 norm
    totalStats->ComputeAverage();
    totalStats->l2norm_ = std::sqrt(totalStats->sumOfSquares_);

    return totalStats; // std::unique_ptr 支持派生类到基类的隐式转换
}

std::unordered_map<Mki::TensorDType,
    std::function<std::unique_ptr<LLM::StatisticsBase>(const void*, size_t)>> typeToFunctionMap = {
    {Mki::TensorDType::TENSOR_DTYPE_INT8,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<int8_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_INT16,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<int16_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_INT32,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<int32_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_INT64,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<int64_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_UINT8,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<uint8_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_UINT16,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<uint16_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_UINT32,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<uint32_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_UINT64,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<uint64_t>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<float>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_DOUBLE,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<double>(binData, dataSize); }},
    {Mki::TensorDType::TENSOR_DTYPE_FLOAT16,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<float>(binData, dataSize,
                Mki::TensorDType::TENSOR_DTYPE_FLOAT16); }},
    {Mki::TensorDType::TENSOR_DTYPE_BF16,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<float>(binData, dataSize,
                Mki::TensorDType::TENSOR_DTYPE_BF16); }},
    {Mki::TensorDType::TENSOR_DTYPE_COMPLEX64,
        [](const void* binData, size_t dataSize)
            { return GetStatisticsFromBinaryDataWithBasicType<std::complex<float>>(binData, dataSize); }}
};

std::unique_ptr<LLM::StatisticsBase> GetStatisticsFromBinaryDataWithTensorDType(
    const void* binData, size_t dataSize, Mki::TensorDType dType)
{
    auto it = typeToFunctionMap.find(dType);
    if (Likely(it != typeToFunctionMap.end())) {
        return it->second(binData, dataSize);
    } else {
        AIT_LOG_WARNING("Unsupported tensor datatype: " + Mki::GetStrWithDType(dType));
        return std::make_unique<LLM::Statistics<std::string>>();
    }
}

static bool IsSaveTensorStats()
{
    const char* saveTensor = std::getenv("ATB_SAVE_TENSOR_STATISTICS");
    if (saveTensor != nullptr) {
        int value = SafetyStoi(saveTensor, 0);
        if (value == atb::SAVE_TENSOR_DATA) {
            return true;
        }
    }
    return false;
}

static bool IsSaveTensorValid(BinFileInfo &inputFile, std::string &outPath)
{
    // 判断是否需要保存
    bool saveFlag = (IsInTensorBinPath(inputFile.filePath) && atb::Probe::IsSaveIntensor()) ||
                    (IsOutTensorBinPath(inputFile.filePath) && atb::Probe::IsSaveOuttensor());
    AIT_LOG_DEBUG("saveFlag: " + std::to_string(saveFlag));
    AIT_LOG_DEBUG("filePath: " + inputFile.filePath);
    if (!saveFlag) { return false; }
    // 检查dataSize是否超过最大值，是否为0
    if ((inputFile.dataSize > MsConst::MAX_FILE_SIZE_DEFAULT) ||
        (inputFile.dataSize == 0)) {
        AIT_LOG_ERROR("Invalid dataSize: " + std::to_string(inputFile.dataSize));
        return false;
    }

    const std::string& outDir = GetOutDir();
    if (outDir == "") { return false; }
    outPath = outDir + ARGS_DUMP_TYPE_TENSOR + "/" + inputFile.filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);
    bool envValidFlag = (IsDeviceIdValid(inputFile.filePath)) && Utils::CheckDirectory(directory) &&
                        (IsDiskSpaceValid(outDir, inputFile.dataSize));
    if (!envValidFlag) { return false; }
    if (!inputFile.hostData) {
        AIT_LOG_ERROR("hostData is None.");
        return false;
    }

    const char* saveTensorInBeforeOutAfter = std::getenv("ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER");
    if (saveTensorInBeforeOutAfter &&
        SafetyStoi(saveTensorInBeforeOutAfter, 0) == SAVE_TENSOR_IN_BEFORE_OUT_AFTER) {
        bool isIntensorBefore = IsSubString(inputFile.filePath, {"before", "intensor"});
        bool isOuttensorAfter = IsSubString(inputFile.filePath, {"after", "outtensor"});
        if (!(isIntensorBefore || isOuttensorAfter)) {
            return false;
        }
    }
    if (!IsTensorFileHeadVaild(inputFile.format) ||
        !IsTensorFileHeadVaild(inputFile.dtype) ||
        !IsTensorFileHeadVaild(inputFile.dims)) {
        return false;
    }
    return true;
}

static inline Mki::TensorDType GetDTypeWithStr(const std::string &dtype)
{
    // 若ATB侧返回的值不再为枚举对应的数字, 而为字符串, 则此处仅保留Mki::GetDTypeWithStr即可
    return Mki::GetDTypeWithStr(Mki::GetStrWithDType(SafetyStoi(dtype.c_str(), -1)));
}

void atb::Probe::SaveTensor(const std::string &format, const std::string &dtype,
    const std::string &dims, const void *hostData, uint64_t dataSize,
    const std::string &filePath)
{
    BinFileInfo inputFile{format, dtype, dims, hostData, dataSize, filePath};
    std::string outPath;
    if (!IsSaveTensorValid(inputFile, outPath)) { return; }
    std::shared_ptr<FileSystem::BinFile> binfile(std::make_shared<FileSystem::BinFile>());
    binfile->AddAttr("format", format);
    binfile->AddAttr("dtype", dtype);
    binfile->AddAttr("dims", dims);
    if (IsSaveTensorStats()) {
        auto stats = GetStatisticsFromBinaryDataWithTensorDType(hostData, dataSize, GetDTypeWithStr(dtype));
        binfile->AddAttr("max", stats ? stats->GetMaxStr() : "N/A");
        binfile->AddAttr("min", stats ? stats->GetMinStr() : "N/A");
        binfile->AddAttr("mean", stats ? stats->GetMeanStr() : "N/A");
        binfile->AddAttr("l2norm", stats ? stats->GetL2NormStr() : "N/A");
    } else if (atb::Probe::IsSaveTensorData()) {
        binfile->AddObject("data", hostData, dataSize);
    }
    GetGlobalPool()->Enqueue(CheckAndWriteFile, binfile, outPath);
}

void atb::Probe::SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath)
{
    if (!data) {
        AIT_LOG_ERROR("Data is None.");
        return;
    }
    // 检查dataSize是否超过最大值，是否为0
    if ((dataSize > MsConst::MAX_FILE_SIZE_DEFAULT) || (dataSize == 0)) {
        AIT_LOG_ERROR("Invalid dataSize: " + dataSize);
        return;
    }
    std::string outDir = GetOutDir();
    std::string outPath = outDir + "tiling/" + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    bool ret = Utils::CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + directory);
        return;
    }

    std::string fileName = outPath.substr(found + 1);
    outPath = GetRealPath(directory);
    ret = Utils::CheckDirectory(outPath);
    if (!ret) {
        AIT_LOG_ERROR("Illegal path: " + outPath);
        return;
    }
    outPath = outPath + "/" + fileName;
    ms::UmaskWrapper uw;
    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(data), dataSize);
        outfile.close();
        AIT_LOG_INFO("Data written to file successfully!");
    } else {
        AIT_LOG_ERROR("Unable to open file! file path: " + outPath);
    }
    return;
}

bool atb::Probe::IsSaveTiling()
{
    const char* isSaveTiling = std::getenv("ATB_SAVE_TILING");
    if (isSaveTiling == nullptr) {
        AIT_LOG_DEBUG("IsSaveTiling: false");
        return false;
    }
    if (SafetyStoi(isSaveTiling, 0) != 0) {
        AIT_LOG_DEBUG("IsSaveTiling: true");
        return true;
    }
    AIT_LOG_DEBUG("IsSaveTiling: false");
    return false;
}

bool atb::Probe::IsSaveIntensor()
{
    const char* saveTensorPart = std::getenv("ATB_SAVE_TENSOR_PART");
    if (saveTensorPart == nullptr) {
        AIT_LOG_DEBUG("IsSaveIntensor: false");
        return false;
    }
    int value = SafetyStoi(saveTensorPart, 0);
    if (value == SAVE_INTENSOR || value == SAVE_ALL_TENSOR) {
        AIT_LOG_DEBUG("IsSaveIntensor: true");
        return true;
    }
    AIT_LOG_DEBUG("IsSaveIntensor: false");
    return false;
}

bool atb::Probe::IsSaveOuttensor()
{
    const char* saveTensorPart = std::getenv("ATB_SAVE_TENSOR_PART");
    if (saveTensorPart == nullptr) {
        AIT_LOG_DEBUG("IsSaveOuttensor: false");
        return false;
    }
    int value = SafetyStoi(saveTensorPart, 0);
    if (value == SAVE_OUTTENSOR || value == SAVE_ALL_TENSOR) {
        AIT_LOG_DEBUG("IsSaveOuttensor: true");
        return true;
    }
    AIT_LOG_DEBUG("IsSaveOuttensor: false");
    return false;
}

bool atb::Probe::ReportOperationGraphEnable()
{
    return IsSaveDumpType("layer") | IsSaveDumpType("model");
}

static void ModifyRootNodeTensors(ordered_json &graphNodeJsonToSave, std::vector<std::string> &tensorNameList,
    const ordered_json &graphNodeJson)
{
    // 根节点根据自己的opName + id, 组成tensor name
    uint32_t inTensorNum = graphNodeJson["inTensorNum"].get<uint32_t>();
    uint32_t outTensorNum = graphNodeJson["outTensorNum"].get<uint32_t>();
    uint32_t internalTensorNum = (graphNodeJson.find("internalTensorNum") == graphNodeJson.end()) ?
                                        0 : graphNodeJson["internalTensorNum"].get<uint32_t>();
    std::string opNameInJson = graphNodeJson["opName"].get<std::string>();

    std::string tensorName;

    for (size_t i = 0; i < inTensorNum; i++) {
        tensorName = opNameInJson + "_input_" + std::to_string(i);
        graphNodeJsonToSave["inTensors"].emplace_back(tensorName);
        tensorNameList.emplace_back(tensorName);
    }
    for (size_t i = 0; i < outTensorNum; i++) {
        tensorName = opNameInJson + "_output_" + std::to_string(i);
        graphNodeJsonToSave["outTensors"].emplace_back(tensorName);
        tensorNameList.emplace_back(tensorName);
    }
    for (size_t i = 0; i < internalTensorNum; i++) {
        tensorName = opNameInJson + "_internal_" + std::to_string(i);
        graphNodeJsonToSave["internalTensors"].emplace_back(tensorName);
        tensorNameList.emplace_back(tensorName);
    }

    AIT_LOG_DEBUG("tensorName: " + tensorName);

    return;
}

static bool CheckGraphInputInvalid(const std::string &opName, const ordered_json &graphNodeJson)
{
    if (graphNodeJson.find("opName") == graphNodeJson.end() ||
        graphNodeJson.find("opType") == graphNodeJson.end() ||
        graphNodeJson.find("inTensorNum") == graphNodeJson.end() ||
        graphNodeJson.find("outTensorNum") == graphNodeJson.end()) {
        AIT_LOG_ERROR("json parse error! opName: " + opName);
        return true;
    }

    std::string opNameInJson = graphNodeJson["opName"].get<std::string>();
    if (opNameInJson != opName) {
        AIT_LOG_ERROR("json parse error! opName is not equal opName in json. opName: " + opName +
            ", opNameInJson: " + opNameInJson);
        return true;
    }
    return false;
}

void saveJsonField(const std::string& fieldName, const ordered_json& graphNodeJson, ordered_json& graphNodeJsonToSave)
{
    try {
        if (graphNodeJson.contains(fieldName)) {
            graphNodeJsonToSave[fieldName] = graphNodeJson[fieldName];
        } else {
            AIT_LOG_ERROR(fieldName + " not found in graph.");
            return;
        }
    } catch (const std::exception& e) {
        AIT_LOG_ERROR("An unexpected error occurred: " + std::string(e.what()));
        return;
    }
}

void atb::Probe::ReportOperationGraph(const std::string &opName, const std::string &graph)
{
    ordered_json graphNodeJson;
    if (Utils::SafetyGuard::CheckNormalStr(opName) != SAFETY_RET::SAFE_ERR_NONE) {
        AIT_LOG_ERROR("Check opName string failed!");
        return;
    }
    try {
        graphNodeJson = ordered_json::parse(graph);
    } catch (const ordered_json::parse_error& ex) {
        AIT_LOG_ERROR("json parse error! opName:" + opName);
        AIT_LOG_ERROR("message: " + std::string(ex.what()) + '\n' + "exception id: " + std::to_string(ex.id) + '\n' +
               "byte position of error: " + std::to_string(ex.byte));
        return;
    }
 
    // 检查必选项
    if (CheckGraphInputInvalid(opName, graphNodeJson)) {
        AIT_LOG_ERROR("CheckGraphInput failed: input is invalid.");
        return;
    }
 
    // 保存原始json信息，用于和model拓扑合并成模型的拓扑信息
    if (IsSaveDumpType("model")) {
        AIT_LOG_DEBUG("Save dump type contains `model`: true.");
        g_layerGraphMap.SaveLayerGraph(opName, graph);
    }
 
    ordered_json graphNodeJsonToSave;
    saveJsonField("opName", graphNodeJson, graphNodeJsonToSave);
    saveJsonField("opType", graphNodeJson, graphNodeJsonToSave);
    saveJsonField("param", graphNodeJson, graphNodeJsonToSave);
 
    // 根节点
    std::vector<std::string> tensorNameList;
    try {
        ModifyRootNodeTensors(graphNodeJsonToSave, tensorNameList, graphNodeJson);
    } catch (const std::exception& e) {
        AIT_LOG_ERROR("An unexpected error occurred: "+ std::string(e.what()));
        return;
    }
    // 递归调用获取子节点信息
    if (graphNodeJson.find("nodes") != graphNodeJson.end()) {
        for (auto childNodeInput : graphNodeJson["nodes"]) {
            ordered_json childNodeToSave;
            try {
                DfsToModifyGraphTensors(childNodeToSave, tensorNameList, childNodeInput);
            } catch (const std::exception& e) {
                AIT_LOG_ERROR("An unexpected error occurred: "+ std::string(e.what()));
                return;
            }
            graphNodeJsonToSave["nodes"].emplace_back(childNodeToSave);
        }
    }
 
    // 保存修改的Json
    const std::string& outDir = GetOutDir();
    if (outDir == "") {
        return;
    }
    std::string pidDir = outDir + "layer/" + std::to_string(GetCurrentProcessId());
    if (!Utils::CheckDirectory(pidDir)) {
        AIT_LOG_ERROR("Create directory failed: " + pidDir);
        return;
    }
 
    std::string outPath = GetRealPath(pidDir);
    bool ret = Utils::CheckDirectory(outPath);
    if (!ret) {
        AIT_LOG_ERROR("Illegal path: " + outPath);
        return;
    }
    outPath = outPath + "/" + opName + ".json";
    ms::UmaskWrapper uw;
    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);
    if (outfile.is_open()) {
        outfile << graphNodeJsonToSave.dump() << std::endl;
        outfile.close();
        AIT_LOG_INFO("layer topo info written to file successfully! File name:" + outPath);
    } else {
        AIT_LOG_ERROR("Unable to open file! File name:" + outPath);
    }
 
    if (IsSaveDumpType("onnx")) {
        SaveSubProcessInfo(outPath);
    }
    return;
}

bool atb::Probe::ReportOperationStatisticEnable()
{
    return IsSaveDumpType("cpu_profiling");
}

void atb::Probe::ReportOperationSetupStatistic(const uint64_t executeCount,
    const std::string &opname, const std::string &st)
{
    // 得到文件保存地址
    const std::string& outDir = GetOutDir();
    if (outDir == "") {
        return;
    }
    std::string filePath = ARGS_DUMP_TYPE_CPU_PROFILING + "/" + std::to_string(GetCurrentProcessId()) + \
                            "/operation_statistic_" + std::to_string(executeCount) + ".txt";
    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    AIT_LOG_DEBUG("outPath: " + outPath);
    AIT_LOG_DEBUG("directory: " + directory);

    // 检验地址是否存在
    bool ret = Utils::CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + directory);
        return;
    }

    outPath = GetRealPath(outPath);
    ms::UmaskWrapper uw;
    std::ofstream file(outPath, std::ios_base::app);
    if (file.is_open()) {
        file << "[" << opname << "]:" << st << std::endl;
        file.close();
    } else {
        AIT_LOG_ERROR("Unable to open file!");
    }
    return;
}

void atb::Probe::ReportOperationExecuteStatistic(const uint64_t executeCount,
    const std::string &opname, const std::string &st)
{
    // 得到文件保存地址
    const std::string& outDir = GetOutDir();
    if (outDir == "") {
        return;
    }
    std::string filePath = ARGS_DUMP_TYPE_CPU_PROFILING + "/" + std::to_string(GetCurrentProcessId()) + \
                            "/operation_statistic_" + std::to_string(executeCount) + ".txt";
    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    AIT_LOG_DEBUG("outPath: " + outPath);
    AIT_LOG_DEBUG("directory: " + directory);

    // 检验地址是否存在
    bool ret = Utils::CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + directory);
        return;
    }

    outPath = GetRealPath(outPath);
    ms::UmaskWrapper uw;
    std::ofstream file(outPath, std::ios_base::app);
    if (file.is_open()) {
        file << "[" << opname << "]:" << st << std::endl;
        file.close();
    } else {
        AIT_LOG_ERROR("Unable to open file!");
    }
    return;
}

static std::string MakeAbsolutePath(const std::string& path)
{
    // 如果传入的是绝对路径，则直接返回路径，如果传入的是相对路径，转化为当前程序运行所在的绝对路径
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    std::string curAbsolutePath = std::string(cwd);
    if (path.empty()) {
        return curAbsolutePath + "/";
    } else if (path[0] == '/') {
        return path;
    } else if (path[0] == '~') {
        const char* curHomePath = std::getenv("HOME");
        std::string expandedPath = curHomePath ? (curHomePath + path.substr(1)) : path;
        return expandedPath;
    } else if (path == "." || path == "./") {
        return curAbsolutePath + "/";
    } else if (path.size() > 1 && path[0] == '.' && path[1] == '/') {
        return curAbsolutePath + path.substr(1);
    }
    return curAbsolutePath + "/" + path;
}

static std::string GetInputString(int &caseNum, const std::string &opName, const std::string &opParam,
    const std::vector<atb::Probe::Tensor> &inTensors, const std::vector<atb::Probe::Tensor> &outTensors)
{
    const std::string& outDir = GetOutDir();
    std::string dumpTensorOutPath = MakeAbsolutePath(outDir) + ARGS_DUMP_TYPE_TENSOR + "/";
    const std::string caseName = opName + std::to_string(caseNum);

    AIT_LOG_DEBUG("caseName: " + caseName);

    int inNum = static_cast<int>(inTensors.size());
    std::string inDType = "";
    std::string inFormat = "";
    std::string inShape = "";
    std::string inPath = "";
    std::string dataGenType = "";
    for (int i = 0; i < inNum; ++i) {
        if (i == inNum - 1) {
            inDType = inDType + inTensors[i].dype;
            inFormat = inFormat + inTensors[i].format;
            inShape = inShape + inTensors[i].shape;
            inPath = inPath + dumpTensorOutPath + inTensors[i].path;
            dataGenType = dataGenType + "customize";
        } else {
            inDType = inDType + inTensors[i].dype + ";";
            inFormat = inFormat + inTensors[i].format + ";";
            inShape = inShape + inTensors[i].shape + ";";
            inPath = inPath + dumpTensorOutPath + inTensors[i].path + ";";
            dataGenType = dataGenType + "customize;";
        }
    }
    int outNum = static_cast<int>(outTensors.size());
    std::string outDType = "";
    std::string outFormat = "";
    std::string outShape = "";
    std::string outPath = "";
    for (int i = 0; i < outNum; ++i) {
        if (i == outNum - 1) {
            outDType = outDType + outTensors[i].dype;
            outFormat = outFormat + outTensors[i].format;
            outShape = outShape + outTensors[i].shape;
            outPath = outPath + dumpTensorOutPath + outTensors[i].path;
        } else {
            outDType = outDType + outTensors[i].dype + ";";
            outFormat = outFormat + outTensors[i].format + ";";
            outShape = outShape + outTensors[i].shape + ";";
            outPath = outPath + dumpTensorOutPath + outTensors[i].path + ";";
        }
    }

    // csv相关校验
    const std::vector<std::string> inputStringParts = {
        caseName, opName, opParam, inDType, inFormat, inShape, outDType, outFormat, outShape, dataGenType,
        inPath, outPath
    };

    for (const std::string& value : inputStringParts) {
        bool ret = Utils::ValidateCsvString(value);
        if (!ret) {
            AIT_LOG_ERROR("Check input string failed! Cannot write into csv!");
            return "";
        }
    }

    const std::string inputString = std::to_string(caseNum) + "|" + caseName + "|" + opName + "|" + opParam + "|" + \
        std::to_string(inNum) + "|" + inDType + "|" + inFormat + "|" + inShape + "|" + \
        std::to_string(outNum) + "|" + outDType + "|" + outFormat + "|" + outShape + "|" + dataGenType + \
        "| |" + inPath + "|" + outPath + "| | | | |NO_ERROR";

    AIT_LOG_DEBUG("inputString: " + inputString);

    return inputString;
}

static void ReportIOTensor(std::string &outPath, const std::string &opName, const std::string &opParam,
    const std::vector<atb::Probe::Tensor> &inTensors, const std::vector<atb::Probe::Tensor> &outTensors)
{
    int caseNum = 1;
    outPath = GetRealPath(outPath);
    ms::UmaskWrapper uw;

    std::ifstream f(outPath, std::ios::in);
    if (f.is_open()) {
        caseNum = 0;
        std::string line;
        const int maxLoopCount = 10000;
        while (std::getline(f, line, '\n') && caseNum < maxLoopCount) {
            caseNum++;
        }
        AIT_LOG_DEBUG("caseNum: " + std::to_string(caseNum));
        if (caseNum == maxLoopCount) {
            AIT_LOG_WARNING("Contents in csv file reached the maximum size of 10000.");
        }
        f.close();
    } else {
        std::ofstream outfile(outPath, std::ios::out);
        caseNum = 1;
        const std::string csvHead = "CaseNum|CaseName|OpName|OpParam|InNum|InDType|InFormat|InShape|OutNum|OutDType|\
OutFormat|OutShape|DataGenType|DataGenRange|InTensorFile|OutTensorFile|TestType|TestLevel|FromModel|SocVersion|\
ExpectedError";
        if (outfile.is_open()) {
            outfile << csvHead << std::endl;
            outfile.close();
        } else {
            AIT_LOG_ERROR("Unable to open file: " + outPath);
        }
    }

    const std::string inputString = GetInputString(caseNum, opName, opParam, inTensors, outTensors);
    std::ofstream outfile(outPath, std::ios::app);
    if (outfile.is_open()) {
        outfile << inputString << std::endl;
        outfile.close();
    } else {
        AIT_LOG_ERROR("Unable to open file: " + outPath);
    }
    return;
}

bool atb::Probe::ReportOperationIOTensorEnable()
{
    return IsSaveDumpType("op");
}

void atb::Probe::ReportOperationIOTensor(const size_t executeCount, const std::string &opName,
    const std::string &opParam, const std::vector<atb::Probe::Tensor> &inTensors,
    const std::vector<atb::Probe::Tensor> &outTensors)
{
    const std::string& outDir = GetOutDir();
    if (outDir == "") {
        return;
    }
    const std::string pid = std::to_string(GetCurrentProcessId());
    std::string fPath = "operation_io_tensors/" + pid + "/operation_tensors_" + std::to_string(executeCount) + ".csv";
    std::string outPath = outDir + fPath;
    size_t found = outPath.find_last_of("/");
    if (found == std::string::npos) {
        std::string errMsg = "Could not find last '/' of outPath, it may caused by process exited abnormally."
                              "Please try again.";
        AIT_LOG_ERROR(errMsg);
        return;
    }
    std::string directory = outPath.substr(0, found);

    bool ret = Utils::CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + directory);
        return;
    }

    ReportIOTensor(outPath, opName, opParam, inTensors, outTensors);
    return;
}

bool atb::Probe::ReportKernelIOTensorEnable()
{
    return IsSaveDumpType("kernel");
}

void atb::Probe::ReportKernelIOTensor(const size_t executeCount, const std::string &opName,
    const std::string &opParam, const std::vector<atb::Probe::Tensor> &inTensors,
    const std::vector<atb::Probe::Tensor> &outTensors)
{
    const std::string& outDir = GetOutDir();
    if (outDir == "") {
        return;
    }
    const std::string pid = std::to_string(GetCurrentProcessId());
    std::string fPath = "kernel_io_tensors/" + pid + "/kernel_tensors_" + std::to_string(executeCount) + ".csv";
    std::string outPath = outDir + fPath;
    size_t found = outPath.find_last_of("/");
    if (found == std::string::npos) {
        AIT_LOG_ERROR("Could not find last / of outPath: " + outPath);
        return;
    }
    std::string directory = outPath.substr(0, found);

    bool ret = Utils::CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + directory);
        return;
    }

    ReportIOTensor(outPath, opName, opParam, inTensors, outTensors);
    return;
}

void atb::Probe::SaveParam(const std::string &param, const std::string &filePath)
{
    const std::string& outDir = GetOutDir();
    if (outDir == "" || !IsDeviceIdValid(filePath)) {
        return;
    }
    std::string outPath = outDir + ARGS_DUMP_TYPE_TENSOR + "/" + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    bool ret = Utils::CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + directory);
        return;
    }

    std::string fileName = outPath.substr(found + 1);
    outPath = GetRealPath(directory);
    ret = Utils::CheckDirectory(outPath);
    if (!ret) {
        AIT_LOG_ERROR("Illegal path: " + outPath);
        return;
    }
    outPath = outPath + "/" + fileName;
    ms::UmaskWrapper uw;
    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);
    if (outfile.is_open()) {
        outfile << param << std::endl;
        outfile.close();
    } else {
        AIT_LOG_ERROR("Unable to open file! File name: " + outPath);
    }
    return;
}

bool atb::Probe::IsSaveParam()
{
    // atb侧该函数返回fasle，通过ait启动模型时改成true，该函数有用，不要删除
    return true;
}

/****************************************************************************************\
                                    算子溢出检测 AIT 接口
\****************************************************************************************/

bool atb::Probe::IsOverflowCheck()
{
    const char* checkType = std::getenv("ATB_CHECK_TYPE");
    if (!checkType) {
        return false;
    }

    bool res = std::string(checkType).find("1") != std::string::npos;
    AIT_LOG_DEBUG("Overflow Check enabled: " + res);

    return res;
}

bool atb::Probe::IsOverflowStop()
{
    const char* exitFlag = std::getenv("ATB_EXIT");
    if (!exitFlag) {
        return false;
    }

    bool res = std::string(exitFlag) == "1";
    AIT_LOG_DEBUG("Terminate after detecting overflow: " + res);

    return res;
}

void atb::Probe::ReportOverflowKernel(const std::string &kernelPath)
{
    if (kernelPath.empty()) {
        return;
    }

    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    if (!outputDir) {
        return;
    }

    char realOutputDir[PATH_MAX] = {0};
    if (realpath(outputDir, realOutputDir) == nullptr) {
        AIT_LOG_ERROR("There is something wrong with the directory, please try another one instead.");
        return;
    }

    if (!Utils::CheckDirectory(realOutputDir)) {
        return;
    }

    const std::string pidID = std::to_string(GetCurrentProcessId());
    const std::string fileName = "msit_overflow_res_" + pidID + ".txt";
    const std::string outPath = std::string(realOutputDir) + "/" + fileName;

    int fd = open(outPath.c_str(), O_CREAT | O_WRONLY, 0600);
    if (fd == -1) {
        AIT_LOG_ERROR("open: " + Utils::GetLastErrorStr());
        AIT_LOG_ERROR("Unable to create file: " + outPath + ". Please check if the output path is valid.");
        return;
    }

    std::FILE *fp = fdopen(fd, "w");
    if (fp == nullptr) {
        AIT_LOG_ERROR("fdopen: " + Utils::GetLastErrorStr());
        close(fd);
        return;
    }

    std::string errMsg = "Overflow detected! Operator name: " + kernelPath;
    int returnCode = std::fputs(errMsg.c_str(), fp);
    if (returnCode == EOF) {
        AIT_LOG_ERROR("fputs: " + Utils::GetLastErrorStr());
        std::fclose(fp);
        return;
    }

    std::fclose(fp);
    return;
}
} // end of namespace atb

namespace atb_speed {
struct ModelGraphMap {
    std::map<std::string, std::string> modelGraphMap_;

    bool IsInitModelGraph(const std::string &modelName)
    {
        auto it = modelGraphMap_.find(modelName);
        return (it == modelGraphMap_.end()) ? true : false;
    };

    void SaveModelGraph(const std::string &modelName, const std::string &graph)
    {
        modelGraphMap_[modelName] = graph;
    };
};
ModelGraphMap g_modelGraphMap;

bool atb_speed::SpeedProbe::IsReportModelTopoInfo(const std::string &modelName)
{
    // 只保存一次
    if (Utils::SafetyGuard::CheckNormalStr(modelName) != SAFETY_RET::SAFE_ERR_NONE)  {
        AIT_LOG_ERROR("Check modelName string failed!");
        return false;
    }
    return IsSaveDumpType("model") && (g_modelGraphMap.IsInitModelGraph(modelName));
}

void atb_speed::SpeedProbe::ReportModelTopoInfo(const std::string &modelName, const std::string &graph)
{
    ordered_json modelJson;
    if (Utils::SafetyGuard::CheckNormalStr(modelName) != SAFETY_RET::SAFE_ERR_NONE) {
        AIT_LOG_ERROR("Check modelName string failed!");
        return;
    }
    g_modelGraphMap.SaveModelGraph(modelName, graph);

    try {
        modelJson = ordered_json::parse(graph);
    } catch (ordered_json::parse_error &ex) {
        AIT_LOG_ERROR("parse model topo info error! modelName: " + modelName);
        AIT_LOG_ERROR("message: " + std::string(ex.what()) + "\nexception id: " + std::to_string(ex.id) +
            "\nbyte position of error: " + std::to_string(ex.byte));
        return;
    }

    // 和atb保存的layer拓扑信息进行合并
    if (modelJson.find("nodes") != modelJson.end()) {
        for (auto &layerJson : modelJson["nodes"]) {
            try {
                MergeLayerTopoInfo(layerJson);
            } catch (const std::exception& e) {
                AIT_LOG_ERROR("An unexpected error occurred: "+ std::string(e.what()));
                return;
            }
        }
    }

    // 保存合并后的Json
    const std::string& outDir = GetOutDir();
    if (outDir == "") {
        return;
    }
    std::string pid = std::to_string(GetCurrentProcessId());
    std::string pidDir = outDir + "model/" + pid;
    bool ret = Utils::CheckDirectory(pidDir);
    if (!ret) {
        AIT_LOG_ERROR("Create directory failed: " + pidDir);
        return;
    }

    std::string outPath = GetRealPath(pidDir);
    ret = Utils::CheckDirectory(outPath);
    if (!ret) {
        AIT_LOG_ERROR("Illegal path: " + outPath);
        return;
    }
    outPath = outPath + "/" + modelName + ".json";
    ms::UmaskWrapper uw;
    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile << modelJson.dump() << std::endl;
        outfile.close();
        AIT_LOG_INFO("model topo info written to file successfully! File name: " + outPath);
    } else {
        AIT_LOG_ERROR("Unable to open file! File name: " + outPath);
    }

    if (IsSaveDumpType("onnx")) {
        SaveSubProcessInfo(outPath);
    }
    return;
}

} // end of namespace atb_speed
