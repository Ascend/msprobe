#!/bin/bash
#Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# ================================================================================
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
