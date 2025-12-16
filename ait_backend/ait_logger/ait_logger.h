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


#ifndef AIT_LOGGER_H
#define AIT_LOGGER_H

#include <cstring>
#include <string>

#define FILENAME (strrchr("/" __FILE__, '/') + 1)
#define AIT_LOG_DEBUG(message) ait::Logger((message), FILENAME, __LINE__, __FUNCTION__, ait::LogLevel::DEBUG)
#define AIT_LOG_INFO(message) ait::Logger((message), FILENAME, __LINE__, __FUNCTION__, ait::LogLevel::INFO)
#define AIT_LOG_WARNING(message) ait::Logger((message), FILENAME, __LINE__, __FUNCTION__, ait::LogLevel::WARNING)
#define AIT_LOG_ERROR(message) ait::Logger((message), FILENAME, __LINE__, __FUNCTION__, ait::LogLevel::ERROR)
#define AIT_LOG_FATAL(message) ait::Logger((message), FILENAME, __LINE__, __FUNCTION__, ait::LogLevel::FATAL)
#define AIT_LOG_CRITICAL(message) ait::Logger((message), FILENAME, __LINE__, __FUNCTION__, ait::LogLevel::CRITICAL)

namespace ait {
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL,
    CRITICAL,
};

void Logger(const std::string message, const char *fileName, int line, const char *funcName, LogLevel level);
}

#endif
