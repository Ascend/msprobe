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