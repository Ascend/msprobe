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
#include "securec.h"

FileSystem::BinFile::BinFile() {}
FileSystem::BinFile::~BinFile() {}

bool FileSystem::BinFile::AddAttr(const std::string &name, const std::string &value)
{
    if (attrNames_.find(name) != attrNames_.end()) {
        std::cout << "Attr: " << name << " already exists" << std::endl;
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
        std::cout << "File to write can't open : " << filePath << std::endl;
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
        std::cout << "binary buffer size is none" << std::endl;
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
            std::cout << "memcpy_s failed, err = " << err << std::endl;
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
