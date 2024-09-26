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

#include "bin_file.h"
#include <sstream>
#include "securec.h"
#include "ait_logger.h"

FileSystem::BinFile::BinFile() {}
FileSystem::BinFile::~BinFile() {}

bool FileSystem::BinFile::AddAttr(const std::string &name, const std::string &value)
{
    if (attrNames_.find(name) != attrNames_.end()) {
        AIT_LOG_ERROR("Attr: " + name + " already exists");
        return false;
    }
    attrNames_.insert(name);
    attrs_.push_back({name, value});

    return true;
}

bool FileSystem::BinFile::Write(const std::string &filePath, const mode_t mode)
{
    // 先写头
    // 先写version、count、length
    // 写format dtype dims
    // 再写data
    // 再写end
    std::ofstream outputFile(filePath, std::ios::app);
    if (!outputFile.is_open()) {
        AIT_LOG_ERROR("File to write can't open : " + filePath);
        return false;
    }

    bool ret = WriteAttr(outputFile, ATTR_VERSION, version_);
    ret = WriteAttr(outputFile, ATTR_OBJECT_COUNT, std::to_string(binaries_.size()));
    ret = WriteAttr(outputFile, ATTR_OBJECT_LENGTH, std::to_string(binariesBuffer_.size()));

    for (const auto &attrIt : attrs_) {
        ret = WriteAttr(outputFile, attrIt.first, attrIt.second);
    }

    for (const auto &objIt : binaries_) {
        ret = WriteAttr(outputFile, ATTR_OBJECT_PREFIX + objIt.first,
                        std::to_string(objIt.second.offset) + "," + std::to_string(objIt.second.length));
    }

    ret = WriteAttr(outputFile, ATTR_END, END_VALUE);

    if (binariesBuffer_.size() > 0U) {
        outputFile.write(binariesBuffer_.data(), binariesBuffer_.size());
    }
    outputFile.close();
    return true;
}

bool FileSystem::BinFile::AddObject(const std::string &name, const void* binaryBuffer, uint64_t binaryLen)
{
    if (binaryBuffer == nullptr) {
        AIT_LOG_ERROR("binary buffer size is none");
        return false;
    }
    size_t needLen = binariesBuffer_.size() + binaryLen;

    if (binaryNames_.find(name) != binaryNames_.end()) {
        return false;
    }

    binaryNames_.insert(name);

    size_t currentLen = binariesBuffer_.size();
    BinFile::Binary binary;
    binary.offset = currentLen;
    binary.length = binaryLen;
    binaries_.push_back({name, binary});
    binariesBuffer_.resize(needLen);

    uint64_t offset = 0;
    uint64_t copyLen = binaryLen;
    while (copyLen > 0) {
        uint64_t curCopySize = copyLen > MAX_SINGLE_MEMCPY_SIZE ? MAX_SINGLE_MEMCPY_SIZE : copyLen;
        auto err = memcpy_s(binariesBuffer_.data() + currentLen + offset, curCopySize,
            static_cast<const uint8_t*>(binaryBuffer) + offset, curCopySize);
        if (err != EOK) {
            AIT_LOG_ERROR("memcpy_s failed, err = " + std::to_string(static_cast<int>(err)));
            return false;
        }
        offset += curCopySize;
        copyLen -= curCopySize;
    }
    return true;
}

bool FileSystem::BinFile::WriteAttr(std::ofstream &outputFile, const std::string &filePath, const std::string &value)
{
    std::string line = filePath + "=" + value + "\n";
    outputFile << line;
    return true;
}

uint64_t FileSystem::BinFile::CalcHash()
{
    const uint64_t fnvOffsetBasis = 14695981039346656037ULL;
    const uint64_t fnvPrime = 1099511628211ULL;
    const int width = 8;

    uint64_t hashValue = fnvOffsetBasis;

    // 计算文件地址
    std::string tempAttrs;
    for (const auto &attrIt : attrs_) {
        tempAttrs += attrIt.first;
        tempAttrs += attrIt.second;
    }
    std::vector<char> hashAttrs(tempAttrs.begin(), tempAttrs.end());
    if (hashAttrs.size() % width != 0) {
        auto newsize = hashAttrs.size() + (width - hashAttrs.size() % width);
        hashAttrs.resize(newsize, 0);
    }
    for (size_t i = 0; i < hashAttrs.size(); i += width) {
        auto addr = &(hashAttrs[i]);
        auto *next64Binary = reinterpret_cast<uint64_t *>(addr);
        hashValue ^= (*next64Binary);
        hashValue *= fnvPrime;
    }

    for (const auto &attrIt : binaries_) {
        hashValue ^= attrIt.second.offset;
        hashValue *= fnvPrime;
        hashValue ^= attrIt.second.length;
        hashValue *= fnvPrime;
    }

    // 计算文件内容
    size_t rawSize = binariesBuffer_.size();
    if (binariesBuffer_.size() % width != 0) {
        auto newsize = binariesBuffer_.size() + (width - binariesBuffer_.size() % width);
        binariesBuffer_.resize(newsize, 0);
    }
    for (size_t i = 0; i < binariesBuffer_.size(); i += width) {
        auto addr = &(binariesBuffer_[i]);
        auto *next64Binary = reinterpret_cast<uint64_t *>(addr);
        hashValue ^= (*next64Binary);
        hashValue *= fnvPrime;
    }
    if (binariesBuffer_.size() != rawSize) {
        binariesBuffer_.resize(rawSize);
    }
    return hashValue;
}