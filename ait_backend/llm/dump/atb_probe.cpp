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
#include <unistd.h>
#include <syscall.h>
#include <cctype>
#include <sys/statvfs.h>
#include "bin_file.h"
#include "nlohmann/json.hpp"
#include "ait_logger.h"
#include "atb_probe.h"

using ordered_json = nlohmann::ordered_json;

namespace {
    unsigned long long g_minDiskSpaceFreeSize = 2147483648; // 2G
    constexpr size_t FREE_SIZE_MULTIPLE_OF_DATA_SIZE = 2; // free size至少两倍data size大小
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
}

static int GetFreeSpace(std::string path, unsigned long long *freeSpace)
{
    struct statvfs diskInfo;

    if (statvfs(path.c_str(), &diskInfo) == -1) {
        AIT_LOG_ERROR("statvfs() error");
        return 1;
    }
    *freeSpace = diskInfo.f_bavail * diskInfo.f_bsize;
    return 0;
}

static int32_t GetCurrentProcessId()
{
    int32_t pid = getpid();
    if (pid == -1) {
        AIT_LOG_WARNING("get pid failed");
    }
    return pid;
}

static bool IsPrefix(const std::string &str, const std::string &prefix)
{
    return str.compare(0, prefix.length(), prefix) == 0;
}

static std::vector<std::string> SplitString(const std::string &ss, const char &tar)
{
    std::vector<std::string> tokens;
    std::stringstream input(ss);
    std::string token;
    while (std::getline(input, token, tar)) {
        tokens.emplace_back(token);
    }

    return tokens;
}


static bool DirectoryExists(const std::string &path)
{
    struct stat info;
    return (stat(path.c_str(), &info) == 0) && (S_ISDIR(info.st_mode));
}

static bool CheckDirectory(const std::string &directory)
{
    std::vector<std::string> dirs = SplitString(directory, '/');
    std::string curDir = "";
    // 检查目录是否存在，如果不存在则创建目录和文件
    for (auto &dir : dirs) {
        curDir += dir + "/";
        if (!DirectoryExists(curDir)) {
            int status = mkdir(curDir.c_str(), 0750);
            if (status) {
                AIT_LOG_WARNING("cannot create directory: " + curDir);
            }
        }
    }
    if (!DirectoryExists(directory)) {
        AIT_LOG_WARNING("cannot create directory: " + directory);
        return false;
    }
    return true;
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
        std::vector<std::string> dumpTypes = SplitString(dumpTypeList, '|');
        for (const auto &type : dumpTypes) {
            if (type == tar) {
                return true;
            }
        }
    }
    return false;
}

static void DfsToModifyGraphTensors(ordered_json &curNodeToSave,
    const std::vector<std::string> &fatherNodeTensorNameList, const ordered_json &curNodeInput)
{
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
            DfsToModifyGraphTensors(childNodeToSave, curNodeTensorNameList, childNodeInput);
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

    // inTensor和outTensor从model里获取，internalTensor自己申请id, 组成layerTensorNameList
    ordered_json atbLayerJson = ordered_json::parse(atbLayerGraph);
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
    bool ret = CheckDirectory(outDir);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + outDir);
        return;
    }

    std::string outPath = outDir + "/subprocess_info.txt";
    std::ofstream outfile(outPath, std::ios::app);

    if (outfile.is_open()) {
        outfile << infoToSave << std::endl;
        outfile.close();
    }
    return;
}

namespace atb {

bool atb::Probe::IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &optype)
{
    if (!IsSaveDumpType("tensor")) {
        return false;
    }

    const char *vid = std::getenv("ATB_SAVE_TENSOR_IDS"); // 应该是20_1_9,1_23,5_29_1
    const char *tid = std::getenv("ATB_SAVE_TENSOR_RUNNER"); // 应该是LinearOps，SelfAttention
    if (!vid && !tid) {
        return true;
    }

    if (vid != nullptr) {
        std::vector<std::string> splitVid = SplitString(vid, ',');
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

    std::string copyOptype = optype;
    for (char &c : copyOptype) {
        c = std::tolower(c);
    }
    // 先用逗号分隔vid和tid

    if (tid != nullptr) {
        std::vector<std::string> splitTid = SplitString(tid, ',');
        for (const auto &indice : splitTid) {
            if (IsPrefix(copyOptype, indice)) {
                return true;
            }
        }
    }

    return false;
}

bool atb::Probe::IsSaveChild()
{
    const char* child = std::getenv("ATB_SAVE_CHILD");
    if (child == nullptr) {
        return false;
    }
    int value = std::stoi(child);
    return value;
}

bool atb::Probe::IsSaveTensorData()
{
    const char* saveTensor = std::getenv("ATB_SAVE_TENSOR");
    if (saveTensor != nullptr) {
        int value = std::stoi(saveTensor);
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
    const char* saveTensorRange = std::getenv("ATB_SAVE_TENSOR_RANGE");
    // overflow check required
    if (!saveTensorRange) {
        return false;
    }
    std::vector<std::string> saveTensorRan = SplitString(saveTensorRange, ',');
    for (size_t i = 1U; i < saveTensorRan.size(); i += RANGE_COUNT) {
        uint64_t left = stoi(saveTensorRan[i - 1]);
        uint64_t right = stoi(saveTensorRan[i]);
        if ((executeCount <= right) && (executeCount >= left)) {
            return true;
        }
    }
    return false;
}


bool atb::Probe::IsSaveTensorBefore()
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    int value = SAVE_TENSOR_AFTER;  // Default to SAVE_TENSOR_AFTER
    if (saveTensorTime) {
        value = std::stoi(saveTensorTime);
    }
    if (value == SAVE_TENSOR_BEFORE || value == SAVE_TENSOR_BOTH) {
        return true;
    }
    return false;
}


bool atb::Probe::IsSaveTensorAfter()
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    int value = SAVE_TENSOR_AFTER;  // Default to SAVE_TENSOR_AFTER
    if (saveTensorTime) {
        value = std::stoi(saveTensorTime);
    }
    if (value == SAVE_TENSOR_AFTER || value == SAVE_TENSOR_BOTH) {
        return true;
    }
    return false;
}


void atb::Probe::SaveTensor(const std::string &format, const std::string &dtype,
    const std::string &dims, const void *hostData, uint64_t dataSize,
    const std::string &filePath)
{
    // 判断是否需要保存
    bool saveFlag = (IsInTensorBinPath(filePath) && IsSaveIntensor()) ||
                (IsOutTensorBinPath(filePath) && IsSaveOuttensor());
    AIT_LOG_DEBUG("saveFlag: " + std::to_string(saveFlag));
    AIT_LOG_DEBUG("filePath: " + filePath);
    if (!saveFlag) {
        return;
    }

    const char* saveDeviceId = std::getenv("ATB_DEVICE_ID");
    if (saveDeviceId) {
        size_t found = filePath.find("_");  // filePath like {device_id}_{pid}/xxx/xxx
        std::string curDeviceId = filePath.substr(0, found);
        if (std::string(saveDeviceId) != curDeviceId) {
            AIT_LOG_DEBUG("Skip saving, curDeviceId: " + curDeviceId);
            return;  // if ATB_DEVICE_ID provided and not equal, skip saving
        }
    }

    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = (outputDir != nullptr ? outputDir : "./");
    outDir = outDir + "ait_dump/tensors/";

    // 磁盘空间判断
    unsigned long long freeSpace = 0;
    int retGetFreeSpace = GetFreeSpace(outDir, &freeSpace);
    if (retGetFreeSpace == 0 &&
        (freeSpace <= g_minDiskSpaceFreeSize || freeSpace <= dataSize * FREE_SIZE_MULTIPLE_OF_DATA_SIZE)) {
        AIT_LOG_WARNING("Create directory failed: " + outDir);
        AIT_LOG_WARNING("Disk space is not enough, it's must more than 2G, free size(MB) is: " +
            std::to_string(freeSpace >> 20));
        return;
    }

    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    bool ret = CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + directory);
        return;
    }

    if (!hostData) {
        AIT_LOG_WARNING("hostData is None.");
        return;
    }
    FileSystem::BinFile binFile;
    binFile.AddAttr("format", format);
    binFile.AddAttr("dtype", dtype);
    binFile.AddAttr("dims", dims);
    if (IsSaveTensorData()) {
        binFile.AddObject("data", hostData, dataSize);
    }
    binFile.Write(outPath);
}


void atb::Probe::SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath)
{
    if (!data) {
        AIT_LOG_WARNING("Data is None.");
        return;
    }
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    bool ret = CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + directory);
        return;
    }

    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(data), dataSize);
        outfile.close();
        AIT_LOG_INFO("Data written to file successfully!");
    } else {
        AIT_LOG_WARNING("Unable to open file! file path: " + outPath);
    }
    return;
}


bool atb::Probe::IsSaveTiling()
{
    const char* isSaveTiling = std::getenv("ATB_SAVE_TILING");
    if (isSaveTiling == nullptr) {
        return false;
    }
    if (std::stoi(isSaveTiling) != 0) {
        return true;
    }
    return false;
}


bool atb::Probe::IsSaveIntensor()
{
    const char* saveTensorPart = std::getenv("ATB_SAVE_TENSOR_PART");
    if (saveTensorPart == nullptr) {
        AIT_LOG_DEBUG("IsSaveIntensor: 0");
        return false;
    }
    int value = std::stoi(saveTensorPart);
    if (value == SAVE_INTENSOR || value == SAVE_ALL_TENSOR) {
        AIT_LOG_DEBUG("IsSaveIntensor: 1");
        return true;
    }
    AIT_LOG_DEBUG("IsSaveIntensor: 0");
    return false;
}


bool atb::Probe::IsSaveOuttensor()
{
    const char* saveTensorPart = std::getenv("ATB_SAVE_TENSOR_PART");
    if (saveTensorPart == nullptr) {
        AIT_LOG_DEBUG("IsSaveOuttensor: 0");
        return false;
    }
    int value = std::stoi(saveTensorPart);
    if (value == SAVE_OUTTENSOR || value == SAVE_ALL_TENSOR) {
        AIT_LOG_DEBUG("IsSaveOuttensor: 1");
        return true;
    }
    AIT_LOG_DEBUG("IsSaveOuttensor: 0");
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

    return;
}

static bool CheckGraphInputInvalid(const std::string &opName, const ordered_json &graphNodeJson)
{
    if (graphNodeJson.find("opName") == graphNodeJson.end() ||
        graphNodeJson.find("opType") == graphNodeJson.end() ||
        graphNodeJson.find("inTensorNum") == graphNodeJson.end() ||
        graphNodeJson.find("outTensorNum") == graphNodeJson.end()) {
        AIT_LOG_WARNING("json parse error! opName: " + opName);
        return true;
    }

    std::string opNameInJson = graphNodeJson["opName"].get<std::string>();
    if (opNameInJson != opName) {
        AIT_LOG_WARNING("json parse error! opName is not equal opName in json. opName: " + opName +
            ", opNameInJson: " + opNameInJson);
        return true;
    }
    return false;
}

void atb::Probe::ReportOperationGraph(const std::string &opName, const std::string &graph)
{
    ordered_json graphNodeJson;

    try {
        graphNodeJson = ordered_json::parse(graph);
    } catch (const ordered_json::parse_error& ex) {
        AIT_LOG_WARNING("json parse error! opName:" + opName);
        AIT_LOG_WARNING("message: " + std::string(ex.what()) + '\n' + "exception id: " + std::to_string(ex.id) + '\n' +
               "byte position of error: " + std::to_string(ex.byte));
        return;
    }

    // 检查必选项
    if (CheckGraphInputInvalid(opName, graphNodeJson)) {
        return;
    }

    // 保存原始json信息，用于和model拓扑合并成模型的拓扑信息
    if (IsSaveDumpType("model")) {
        g_layerGraphMap.SaveLayerGraph(opName, graph);
    }

    ordered_json graphNodeJsonToSave;
    graphNodeJsonToSave["opName"] = graphNodeJson["opName"];
    graphNodeJsonToSave["opType"] = graphNodeJson["opType"];
    graphNodeJsonToSave["param"] = graphNodeJson["param"];

    // 根节点
    std::vector<std::string> tensorNameList;
    ModifyRootNodeTensors(graphNodeJsonToSave, tensorNameList, graphNodeJson);

    // 递归调用获取子节点信息
    if (graphNodeJson.find("nodes") != graphNodeJson.end()) {
        for (auto childNodeInput : graphNodeJson["nodes"]) {
            ordered_json childNodeToSave;
            DfsToModifyGraphTensors(childNodeToSave, tensorNameList, childNodeInput);
            graphNodeJsonToSave["nodes"].emplace_back(childNodeToSave);
        }
    }

    // 保存修改的Json
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    std::string pidDir = outDir + "ait_dump/layer/" + std::to_string(GetCurrentProcessId()) + "/";
    bool ret = CheckDirectory(pidDir);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + pidDir);
        return;
    }

    std::string outPath = pidDir + opName + ".json";
    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);
    if (outfile.is_open()) {
        outfile << graphNodeJsonToSave.dump() << std::endl;
        outfile.close();
        AIT_LOG_INFO("layer topo info written to file successfully! File name:" + outPath);
    } else {
        AIT_LOG_WARNING("Unable to open file! File name:" + outPath);
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
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    std::string filePath = "ait_dump/cpu_profiling/" + std::to_string(GetCurrentProcessId()) + \
                            "/operation_statistic_" + std::to_string(executeCount) + ".txt";
    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    // 检验地址是否存在
    bool ret = CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + directory);
        return;
    }

    std::ofstream file(outPath, std::ios_base::app);
    if (file.is_open()) {
        file << "[" << opname << "]:" << st << std::endl;
        file.close();
    } else {
        AIT_LOG_WARNING("Unable to open file!");
    }
    return;
}

void atb::Probe::ReportOperationExecuteStatistic(const uint64_t executeCount,
    const std::string &opname, const std::string &st)
{
    // 得到文件保存地址
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    std::string filePath = "ait_dump/cpu_profiling/" + std::to_string(GetCurrentProcessId()) + \
                            "/operation_statistic_" + std::to_string(executeCount) + ".txt";
    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    // 检验地址是否存在
    bool ret = CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + directory);
        return;
    }

    std::ofstream file(outPath, std::ios_base::app);
    if (file.is_open()) {
        file << "[" << opname << "]:" << st << std::endl;
        file.close();
    } else {
        AIT_LOG_WARNING("Unable to open file!");
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
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    std::string curAbsolutePath = MakeAbsolutePath(outDir);
    std::string dumpTensorOutPath = curAbsolutePath + "ait_dump/tensors/";
    const std::string caseName = opName + std::to_string(caseNum);
    int inNum = inTensors.size();
    std::string inDType = "";
    std::string inFormat = "";
    std::string inShape = "";
    std::string inPath = "";
    for (int i = 0; i < inNum; ++i) {
        if (i == inNum - 1) {
            inDType = inDType + inTensors[i].dype;
            inFormat = inFormat + inTensors[i].format;
            inShape = inShape + inTensors[i].shape;
            inPath = inPath + inTensors[i].path;
        } else {
            inDType = inDType + inTensors[i].dype + ";";
            inFormat = inFormat + inTensors[i].format + ";";
            inShape = inShape + inTensors[i].shape + ";";
            inPath = inPath + inTensors[i].path + ";";
        }
    }
    int outNum = outTensors.size();
    std::string outDType = "";
    std::string outFormat = "";
    std::string outShape = "";
    std::string outPath = "";
    for (int i = 0; i < outNum; ++i) {
        if (i == outNum - 1) {
            outDType = outDType + outTensors[i].dype;
            outFormat = outFormat + outTensors[i].format;
            outShape = outShape + outTensors[i].shape;
            outPath = outPath + outTensors[i].path;
        } else {
            outDType = outDType + outTensors[i].dype + ";";
            outFormat = outFormat + outTensors[i].format + ";";
            outShape = outShape + outTensors[i].shape + ";";
            outPath = outPath + outTensors[i].path + ";";
        }
    }
    const std::string inputString = std::to_string(caseNum) + "|" + caseName + "|" + opName + "|" + opParam + "|" + \
        std::to_string(inNum) + "|" + inDType + "|" + inFormat + "|" + inShape + "|" + \
        std::to_string(outNum) + "|" + outDType + "|" + outFormat + "|" + outShape + "|" + \
        "customize| |" + dumpTensorOutPath + inPath + "|" + dumpTensorOutPath + outPath + "| | | | |NO_ERROR";
    return inputString;
}


static void ReportIOTensor(std::string &outPath, const std::string &opName, const std::string &opParam,
    const std::vector<atb::Probe::Tensor> &inTensors, const std::vector<atb::Probe::Tensor> &outTensors)
{
    int caseNum;
    std::ifstream f(outPath, std::ios::in);
    if (f.is_open()) {
        caseNum = 0;
        std::string line;
        const int maxLoopCount = 10000;
        while (std::getline(f, line, '\n') && caseNum < maxLoopCount) {
            caseNum++;
        }
        if (caseNum == maxLoopCount) {
            AIT_LOG_WARNING("Too many lines in csv file. Maxcount is 10000.");
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
            AIT_LOG_WARNING("Unable to open file: " + outPath);
        }
    }

    const std::string inputString = GetInputString(caseNum, opName, opParam, inTensors, outTensors);
    std::ofstream outfile(outPath, std::ios::app);
    if (outfile.is_open()) {
        outfile << inputString << std::endl;
        outfile.close();
    } else {
        AIT_LOG_WARNING("Unable to open file: " + outPath);
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
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    const std::string pid = std::to_string(GetCurrentProcessId());
    std::string fPath =
        "ait_dump/operation_io_tensors/" + pid + "/operation_tensors_" + std::to_string(executeCount) + ".csv";
    std::string outPath = outDir + fPath;
    size_t found = outPath.find_last_of("/");
    if (found == std::string::npos) {
        AIT_LOG_WARNING("Could not find last / of outPath: " + outPath);
        return;
    }
    std::string directory = outPath.substr(0, found);

    bool ret = CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + directory);
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
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    const std::string pid = std::to_string(GetCurrentProcessId());
    std::string fPath =
        "ait_dump/kernel_io_tensors/" + pid + "/kernel_tensors_" + std::to_string(executeCount) + ".csv";
    std::string outPath = outDir + fPath;
    size_t found = outPath.find_last_of("/");
    if (found == std::string::npos) {
        AIT_LOG_WARNING("Could not find last / of outPath: " + outPath);
        return;
    }
    std::string directory = outPath.substr(0, found);

    bool ret = CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + directory);
        return;
    }

    ReportIOTensor(outPath, opName, opParam, inTensors, outTensors);
    return;
}

void atb::Probe::SaveParam(const std::string &param, const std::string &filePath)
{
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    outDir = outDir + "ait_dump/tensors/";

    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    bool ret = CheckDirectory(directory);
    if (!ret) {
        AIT_LOG_WARNING("[atb::Probe::SaveParam] Create directory failed: " + directory);
        return;
    }

    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);
    if (outfile.is_open()) {
        outfile << param << std::endl;
        outfile.close();
    } else {
        AIT_LOG_WARNING("Unable to open file! File name: " + outPath);
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
    AIT_LOG_DEBUG("IsOverflowCheck is invoked...");

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
    AIT_LOG_DEBUG("IsOverflowStop is invoked...");

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
    AIT_LOG_DEBUG("ReportOverflowKernel is invoked...");

    if (kernelPath.empty()) {
        return;
    }

    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    if (!outputDir) {
        AIT_LOG_WARNING("The environment variable ATB_OUTPUT_DIR is not set.");
        return;
    }

    const std::string pidID = std::to_string(GetCurrentProcessId());
    const std::string fileName = "ait_overflow_res_" + pidID + ".txt";
    const std::string outPath = std::string(outputDir) + "/" + fileName;

    std::ofstream ofs(outPath, std::ios::app);
    if (ofs.is_open()) {
        AIT_LOG_INFO("Output File created. File name: " + outPath);
        ofs << "Overflow detected! Operator name: " << kernelPath << std::endl;
        ofs.close();
    } else {
        AIT_LOG_WARNING("Unable to create file: " + outPath + ". Please check if the directory is valid.");
        ofs.close();
    }

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
    return IsSaveDumpType("model") && (g_modelGraphMap.IsInitModelGraph(modelName));
}

void atb_speed::SpeedProbe::ReportModelTopoInfo(const std::string &modelName, const std::string &graph)
{
    ordered_json modelJson;
    g_modelGraphMap.SaveModelGraph(modelName, graph);

    try {
        modelJson = ordered_json::parse(graph);
    } catch (ordered_json::parse_error &ex) {
        AIT_LOG_WARNING("parse model topo info error! modelName: " + modelName);
        AIT_LOG_WARNING("message: " + std::string(ex.what()) + "\nexception id: " + std::to_string(ex.id) +
            "\nbyte position of error: " + std::to_string(ex.byte));
        return;
    }

    // 和atb保存的layer拓扑信息进行合并
    if (modelJson.find("nodes") != modelJson.end()) {
        for (auto &layerJson : modelJson["nodes"]) {
            MergeLayerTopoInfo(layerJson);
        }
    }

    // 保存合并后的Json
    const char *outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";
    std::string pid = std::to_string(GetCurrentProcessId());
    std::string pidDir = outDir + "ait_dump/model/" + pid + "/";
    bool ret = CheckDirectory(pidDir);
    if (!ret) {
        AIT_LOG_WARNING("Create directory failed: " + pidDir);
        return;
    }

    std::string outPath = pidDir + modelName + ".json";
    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile << modelJson.dump() << std::endl;
        outfile.close();
        AIT_LOG_INFO("model topo info written to file successfully! File name: " + outPath);
    } else {
        AIT_LOG_WARNING("Unable to open file! File name: " + outPath);
    }

    if (IsSaveDumpType("onnx")) {
        SaveSubProcessInfo(outPath);
    }
    return;
}

} // end of namespace atb_speed
