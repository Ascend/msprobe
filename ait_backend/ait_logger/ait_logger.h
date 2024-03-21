/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
#ifndef AIT_LOGGER_H
#define AIT_LOGGER_H

#include <cstring>

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
