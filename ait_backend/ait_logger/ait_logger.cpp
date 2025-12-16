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
        } catch (const std::out_of_range& e) {
            std::cout << "[WARNING] [" << fileName << "+" << line << "][" << funcName << "] "
            << "Cannot convert environment variable to int." << "\n";
            std::cout << "[WARNING] Log level resets to INFO." << std::endl;
        }
    }
    if (atbLogLevel < int(LogLevel::DEBUG) || atbLogLevel > int(LogLevel::CRITICAL)) {
        atbLogLevel = int(LogLevel::INFO);
    }
    int levelInt = int(level);  // levelInt from Enum has to in the range of length of levelName
    if (atbLogLevel <= levelInt) {
        std::cout << "[" << levelName[levelInt] << "] [" << fileName << "+" << line << "][" << funcName << "] "
                  << message << std::endl;
    }
}
