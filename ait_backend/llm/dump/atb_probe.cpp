/*
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
#include "binfile.h"

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
            int status = mkdir(curDir.c_str(), 0755);
            if (status) {
                std::cout << "cannot create directory: " << curDir << std::endl;
            }
        }
    }
    if (!DirectoryExists(directory)) {
        std::cout << "cannot create directory: " << directory << std::endl;
        return false;
    }
    return true;
}


static bool isInTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind("/");
    std::string fileName = filePath;
    if (sepPos != std::string::npos) {
        fileName.erase(0, sepPos + 1U);
    }
    return (fileName.find("intensor") != std::string::npos) || (fileName.find("inTensor") != std::string::npos);
}


static bool isOutTensorBinPath(const std::string &filePath)
{
    size_t sepPos = filePath.rfind("/");
    std::string fileName = filePath;
    if (sepPos != std::string::npos) {
        fileName.erase(0, sepPos + 1U);
    }
    return (fileName.find("outtensor") != std::string::npos) || (fileName.find("outTensor") != std::string::npos);
}


bool atb::Probe::IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &optype)
{
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
    int value = std::stoi(saveTensorTime);
    if (value == SAVE_TENSOR_BEFORE || value == SAVE_TENSOR_BOTH) {
        return true;
    }
    return false;
}


bool atb::Probe::IsSaveTensorAfter()
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    int value = std::stoi(saveTensorTime);
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
    bool saveFlag = (isInTensorBinPath(filePath) && IsSaveIntensor()) ||
                (isOutTensorBinPath(filePath) && IsSaveOuttensor());
    if (!saveFlag) {
        return;
    }

    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = (outputDir != nullptr? outputDir : "./");
    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    bool ret = CheckDirectory(directory);
    if (!ret) {
        std::cout << "Create directory failed: " << directory << std::endl;
        return;
    }

    if (!hostData) {
        std::cout << "hostData is None." << std::endl;
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
        std::cout << "Data is None." << std::endl;
        return;
    }
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr? outputDir : "./";
    std::string outPath = outDir + filePath;
    size_t found = outPath.find_last_of("/");
    std::string directory = outPath.substr(0, found);

    bool ret = CheckDirectory(directory);
    if (!ret) {
        std::cout << "Create directory failed: " << directory << std::endl;
        return;
    }

    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(data), dataSize);
        outfile.close();
        std::cout << "Data written to file successfully!" << std::endl;
    } else {
        std::cout << "Unable to open file! file path: " << outPath << std::endl;
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
        return false;
    }
    int value = std::stoi(saveTensorPart);
    if (value == SAVE_INTENSOR || value == SAVE_ALL_TENSOR) {
        return true;
    }
    return false;
}


bool atb::Probe::IsSaveOuttensor()
{
    const char* saveTensorPart = std::getenv("ATB_SAVE_TENSOR_PART");
    if (saveTensorPart == nullptr) {
        return false;
    }
    int value = std::stoi(saveTensorPart);
    if (value == SAVE_OUTTENSOR || value == SAVE_ALL_TENSOR) {
        return true;
    }
    return false;
}