/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Create Date: 2025
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


#ifndef FILEREGISTRY_FILESTATEGUARD_H
#define FILEREGISTRY_FILESTATEGUARD_H

namespace FileRegistry {

    // 辅助类：保存和恢复文件状态（RAII模式
    class FileStateGuard {
    public:
        explicit FileStateGuard(std::ifstream& file) : file_(file),
            pos_(file.tellg()),
            state_(file.rdstate()) {}
            
        ~FileStateGuard()
        {
            file_.clear();
            file_.seekg(pos_);
            file_.setstate(state_);
        }

    private:
        std::ifstream& file_;
        std::streampos pos_;
        std::ios::iostate state_;
    };

    // 辅助函数：定位数据起始位置
    std::streampos FindDataStart(std::ifstream& inFile);
}

#endif // FILEREGISTRY_FILESTATEGUARD_H