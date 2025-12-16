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


#include "bin_file.h"
#include "securec.h"
#include "umask_wrapper.h"
#include "ait_logger.h"

namespace FileSystem {

BinFile::BinFile() {}
BinFile::~BinFile() {}

bool BinFile::AddAttr(const std::string &name, const std::string &value)
{
    if (attrNames_.find(name) != attrNames_.end()) {
        AIT_LOG_ERROR("Attr: " + name + " already exists");
        return false;
    }
    attrNames_.insert(name);
    attrs_.push_back({name, value});

    return true;
}

bool BinFile::HasAttr(const std::string &name)
{
    return attrNames_.find(name) != attrNames_.end();
}

bool BinFile::Write(const std::string &filePath, const mode_t mode)
{
    // 先写头
    // 先写version、count、length
    // 写format dtype dims
    // 再写data
    // 再写end
    ms::UmaskWrapper um;
    std::ofstream outputFile(filePath, std::ios::app);
    if (!outputFile.is_open()) {
        AIT_LOG_ERROR("File to write can't open : " + filePath);
        return false;
    }

    WriteAttr(outputFile, ATTR_VERSION, version_);
    WriteAttr(outputFile, ATTR_OBJECT_COUNT, std::to_string(binaries_.size()));
    WriteAttr(outputFile, ATTR_OBJECT_LENGTH, std::to_string(binariesBuffer_.size()));

    for (const auto &attrIt : attrs_) {
        WriteAttr(outputFile, attrIt.first, attrIt.second);
    }

    for (const auto &objIt : binaries_) {
        WriteAttr(outputFile, ATTR_OBJECT_PREFIX + objIt.first,
                  std::to_string(objIt.second.offset) + "," + std::to_string(objIt.second.length));
    }

    WriteAttr(outputFile, ATTR_END, END_VALUE);

    if (binariesBuffer_.size() > 0U) {
        outputFile.write(binariesBuffer_.data(), binariesBuffer_.size());
    }
    outputFile.close();
    return true;
}

bool BinFile::AddObject(const std::string &name, const void* binaryBuffer, uint64_t binaryLen)
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

bool BinFile::WriteAttr(std::ofstream &outputFile, const std::string &name, const std::string &value)
{
    std::string line = name + "=" + value + "\n";
    outputFile << line;
    if (!outputFile.good()) {
        AIT_LOG_WARNING("Failed to write " + name + " attribute");
        return false;
    }
    return true;
}
} // end of namespace FileSystem
