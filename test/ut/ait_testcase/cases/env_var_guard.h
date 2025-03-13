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

#ifndef ENV_VAR_GUARD_H
#define ENV_VAR_GUARD_H

#include <string>

/******************​ 环境变量保护工具 ​******************/
class EnvVarGuard {
public:
    explicit EnvVarGuard(const std::string& var) : var_(var)
    {
        auto val = getenv(var.c_str());
        if (val) {
            oldVal_ = val;
        }
    }

    ~EnvVarGuard()
    {
        if (!oldVal_.empty()) {
            setenv(var_.c_str(), oldVal_.c_str(), 1);
        } else {
            unsetenv(var_.c_str());
        }
    }
private:
    std::string var_;
    std::string oldVal_;
};

#endif // end if ENV_VAR_GUARD_H