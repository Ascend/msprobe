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
#include <string>
#include <iostream>
#include "ait_logger.h"


void ait::Logger(const std::string message, const char *fileName, int line, const char *funcName, LogLevel level)
{
    const char* atbLog = std::getenv("ATB_AIT_LOG_LEVEL");
    const char* levelName[] = {"DEBUG", "INFO", "WARNING", "ERROR", "FATAL", "CRITICAL"};
    int atbLogLevel = int(LogLevel::INFO);
    if (atbLog) {
        try {
            atbLogLevel = std::stoi(atbLog);
        } catch (const std::invalid_argument& e) {
            std::cout << "[WARNING] [" << fileName << "+" << line << "][" << funcName << "] "
            << "Cannot convert environment variable to int." << "\n";
            std::cout << "[WARNING] Log level resets to INFO." << std::endl;
        }
    }
    int levelInt = int(level);  // levelInt from Enum has to in the range of length of levelName
    if (atbLogLevel <= levelInt) {
        std::cout << "[" << levelName[levelInt] << "] [" << fileName << "+" << line << "][" << funcName << "] "
                  << message << std::endl;
    }
}
