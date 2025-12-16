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


#ifndef BINFILE_H
#define BINFILE_H

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <set>
#include <map>
#include <sstream>
#include <fstream>
#define EXPORT_LLM __attribute__ ((visibility("default")))

namespace FileSystem {
const std::string ATTR_VERSION = "$Version";
const std::string ATTR_END = "$End";
const std::string ATTR_OBJECT_LENGTH = "$Object.Length";
const std::string ATTR_OBJECT_COUNT = "$Object.Count";
const std::string ATTR_OBJECT_PREFIX = "$Object.";
const std::string END_VALUE = "1";

constexpr mode_t BIN_FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP;
constexpr uint64_t MAX_SINGLE_MEMCPY_SIZE = 1073741824;

class BinFile {
struct Binary {
    uint64_t offset = 0UL;
    uint64_t length = 0UL;
};
public:
    EXPORT_LLM BinFile();
    EXPORT_LLM ~BinFile();

    EXPORT_LLM bool AddAttr(const std::string &name, const std::string &value);
    EXPORT_LLM bool HasAttr(const std::string &name);
    EXPORT_LLM bool Write(const std::string &filePath, const mode_t mode = BIN_FILE_MODE);
    EXPORT_LLM bool WriteAttr(std::ofstream &outputFile, const std::string &name, const std::string &value);
    EXPORT_LLM bool AddObject(const std::string &name, const void* binaryBuffer, uint64_t binaryLen);

private:
    std::string version_ = "1.0";
    std::set<std::string> attrNames_;
    std::vector<std::pair<std::string, std::string>> attrs_;

    std::set<std::string> binaryNames_;
    std::vector<std::pair<std::string, Binary>> binaries_;
    std::vector<char> binariesBuffer_;
};
}
#endif
