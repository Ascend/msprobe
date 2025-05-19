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


#ifndef FILEREGISTRY_FILEREGISTRY_H
#define FILEREGISTRY_FILEREGISTRY_H

#include <mutex>
#include <unordered_map>

namespace FileRegistry {
    class FileRegistry {
    public:
        FileRegistry() { UpdateEnableSymlink(); }
        ~FileRegistry() = default;

        bool GetEnableSymlink() const { return enableSymlink_; }
        const std::string* RegisterFile(const uint64_t hash,
                                        const std::string& path);
        static void UpdateEnableSymlink();

    private:
        std::mutex mtx_;
        std::unordered_map<uint64_t, std::string> registry_;
        static bool enableSymlink_;
    };
}

#endif // FILEREGISTRY_FILEREGISTRY_H