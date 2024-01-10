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


#ifndef ATB_PROBE_H
#define ATB_PROBE_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>


namespace atb {
constexpr int SAVE_TENSOR_BEFORE = 0;
constexpr int SAVE_TENSOR_AFTER = 1;
constexpr int SAVE_TENSOR_BOTH = 2;
constexpr int SAVE_TENSOR_DATA = 1;
constexpr int RANGE_COUNT = 2;
constexpr int SAVE_INTENSOR = 0;
constexpr int SAVE_OUTTENSOR = 1;
constexpr int SAVE_ALL_TENSOR = 2;

class Probe {
public:
    static bool IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &optype);
    static bool IsSaveTensorData();
    static bool IsSaveTensorDesc();
    static bool IsSaveChild();
    static bool IsExecuteCountInRange(const uint64_t executeCount);
    static bool IsSaveTensorBefore();
    static bool IsSaveTensorAfter();
    static void SaveTensor(const std::string &format, const std::string &dtype,
        const std::string &dims, const void *deviceData, uint64_t dataSize,
        const std::string &filePath);
    static void SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath);
    static bool IsSaveTiling();
    static bool IsSaveOuttensor();
    static bool IsSaveIntensor();
};
}
#endif