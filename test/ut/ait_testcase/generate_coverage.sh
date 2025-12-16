#!/bin/bash
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

echo "***************Generate Coverage*****************"

if [ -d "./coverage" ]; then
    rm -rf ./coverage
fi
mkdir coverage

lcov_opt="--rc lcov_branch_coverage=1 --rc geninfo_no_exception_branch=1"
lcov -c -d ./build/CMakeFiles/ait_backend_ut.dir -o ./coverage/ait_backend_ut.info -b ./coverage $lcov_opt

lcov -r ./coverage/ait_backend_ut.info '*platform*' -o ./coverage/ait_backend_ut.info $lcov_opt
lcov -r ./coverage/ait_backend_ut.info '*opensource*' -o ./coverage/ait_backend_ut.info $lcov_opt
lcov -r ./coverage/ait_backend_ut.info '*test*' -o ./coverage/ait_backend_ut.info $lcov_opt
lcov -r ./coverage/ait_backend_ut.info '*c++*' -o ./coverage/ait_backend_ut.info $lcov_opt
lcov -r ./coverage/ait_backend_ut.info '/usr/include/*' -o ./coverage/ait_backend_ut.info $lcov_opt
lcov -r ./coverage/ait_backend_ut.info '*nlohmann*' -o ./coverage/ait_backend_ut.info $lcov_opt

genhtml ./coverage/ait_backend_ut.info -o ./coverage/report --branch-coverage

cd coverage
tar -zcvf report.tar.gz ./report