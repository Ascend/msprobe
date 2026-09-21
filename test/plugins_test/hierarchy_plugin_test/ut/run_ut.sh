# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

#!/bin/bash

# Hierarchy Plugin 单元测试运行脚本
# 功能：运行该目录下所有基于pytest的单元测试用例并生成覆盖率报告

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Hierarchy Plugin 单元测试运行脚本${NC}"
echo -e "${BLUE}================================${NC}"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 创建测试输出目录
TEST_OUTPUT_DIR="$SCRIPT_DIR/test_output"
mkdir -p "$TEST_OUTPUT_DIR"

echo -e "${GREEN}测试目录: $SCRIPT_DIR${NC}"
echo -e "${GREEN}输出目录: $TEST_OUTPUT_DIR${NC}"
echo -e "${GREEN}项目根目录: $PROJECT_ROOT${NC}"

# 清理之前的测试结果
echo -e "${BLUE}清理之前的测试结果...${NC}"
rm -f "$TEST_OUTPUT_DIR"/.coverage*
rm -rf "$TEST_OUTPUT_DIR"/htmlcov*
rm -rf "$TEST_OUTPUT_DIR"/pytest_*

# 运行pytest单元测试并生成覆盖率报告
echo -e "${BLUE}开始运行单元测试...${NC}"

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行测试命令
pytest "$SCRIPT_DIR" \
    --verbose \
    --tb=short \
    --cov=plugins.tb_graph_ascend.hierarchy_plugin \
    --cov-report=term-missing \
    --cov-report=html:"$TEST_OUTPUT_DIR/htmlcov" \
    --cov-report=xml:"$TEST_OUTPUT_DIR/coverage.xml" \
    --junitxml="$TEST_OUTPUT_DIR/test-results.xml" \
    --disable-warnings \
    -o cache_dir="$TEST_OUTPUT_DIR/.pytest_cache" \
    "$@"

TEST_EXIT_CODE=$?

echo -e "${BLUE}================================${NC}"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 所有测试通过!${NC}"
else
    echo -e "${RED}❌ 测试失败，退出码: $TEST_EXIT_CODE${NC}"
fi

# 显示覆盖率报告摘要
if [ -f "$TEST_OUTPUT_DIR/htmlcov/index.html" ]; then
    echo -e "${GREEN}📊 覆盖率报告已生成:${NC}"
    echo -e "${GREEN}   HTML报告: $TEST_OUTPUT_DIR/htmlcov/index.html${NC}"
    echo -e "${GREEN}   XML报告: $TEST_OUTPUT_DIR/coverage.xml${NC}"
fi

if [ -f "$TEST_OUTPUT_DIR/test-results.xml" ]; then
    echo -e "${GREEN}📋 测试结果: $TEST_OUTPUT_DIR/test-results.xml${NC}"
fi

# 显示测试统计
if command -v coverage &> /dev/null; then
    echo -e "${BLUE}覆盖率统计:${NC}"
    coverage report --rcfile="$SCRIPT_DIR/.coveragerc" 2>/dev/null || true
fi

exit $TEST_EXIT_CODE