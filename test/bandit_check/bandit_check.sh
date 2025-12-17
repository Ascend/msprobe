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

src_path=${WORKSPACE}/msfmktransplt/src/
out_path=${WORKSPACE}/bandit_check.html
config_file=${WORKSPACE}/msfmktransplt/test/bandit_check/config.yaml
baseline_file=${WORKSPACE}/msfmktransplt/test/bandit_check/baseline.json
/home/slave1/.local/bin/bandit -c ${config_file} -b ${baseline_file} -r -a file -f html -o ${out_path} ${src_path}
ret=$?
# if command run failed or not
if [ ! -f "${out_path}" ];then
  echo "Bandit run failed."
  exit 1
fi
# command success but if find issues, bandit will return 1
echo "Bandit run success and returns ${ret}."
# upload result file manually
/opt/buildtools/ArtGet_Linux_1.1.8.2/artget push -d ci/JenkinsFile/Build_2.0/bandit/dependency_snapshot.xml -ap "${WORKSPACE}"

exit ${ret}
