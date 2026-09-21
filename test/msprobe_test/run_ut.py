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

import os
import time
import shutil
import subprocess
import sys


class logger:
    def info(self, msg):
        self._print_log("INFO", msg)
    def error(self, msg):
        self._print_log("ERROR", msg)

    def _print_log(self, level, msg, end='\n'):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pid = os.getpid()
        full_msg = f"{current_time} ({pid}) [{level}] {msg}"
        print(full_msg, end=end)
        sys.stdout.flush()


logger = logger()


def _run_pytest(ut_dirs, pythonpath, report_dir, cov_dir, cov_config_path, 
                final_xml_path, html_cov_report, xml_cov_report, test_group_name):
    """
    执行 pytest 测试
    
    Args:
        ut_dirs: 测试目录列表
        pythonpath: PYTHONPATH 环境变量值
        report_dir: 报告目录
        cov_dir: 覆盖率目录
        cov_config_path: 覆盖率配置文件路径
        final_xml_path: 最终 XML 报告路径
        html_cov_report: HTML 覆盖率报告路径
        xml_cov_report: XML 覆盖率报告路径
        test_group_name: 测试组名称（用于日志）
    
    Returns:
        bool: 测试是否成功
    """
    logger.info(f"{test_group_name} UT execution order: {ut_dirs}")
    logger.info(f"{test_group_name} PYTHONPATH: {pythonpath}")
    
    # 为不同测试组生成不同的报告文件
    group_xml_path = os.path.join(report_dir, f"{test_group_name}_junit.xml")
    group_html_cov = os.path.join(report_dir, f"{test_group_name}_htmlcov")
    group_xml_cov = os.path.join(report_dir, f"{test_group_name}_coverage.xml")
    
    pytest_cmd = [
        "python3", "-m", "pytest",
        *ut_dirs,
        f"--junitxml={group_xml_path}",
        f"--cov-config={cov_config_path}",
        f"--cov={cov_dir}",
        "--cov-branch",
        f"--cov-report=html:{group_html_cov}",
        f"--cov-report=xml:{group_xml_cov}",
        "-v"
    ]
    
    # 准备环境变量
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    if pythonpath:
        # 获取原有的 PYTHONPATH（如果有）
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            # 追加到原有 PYTHONPATH 后面
            env["PYTHONPATH"] = existing_pythonpath + os.pathsep + pythonpath
        else:
            # 如果原有 PYTHONPATH 为空，直接设置
            env["PYTHONPATH"] = pythonpath
    # 如果 pythonpath 为空，保持原有的 PYTHONPATH（不做任何操作）
    
    try:
        with subprocess.Popen(
                pytest_cmd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
        ) as proc:
            for line in proc.stdout:
                logger.info(line.strip())
            
            proc.wait()
            
            if proc.returncode == 0:
                logger.info(f"{test_group_name} unit tests executed successfully.")
                return True
            else:
                logger.error(f"{test_group_name} unit tests execution failed.")
                return False
    except Exception as e:
        logger.error(f"An error occurred during {test_group_name} test execution: {e}")
        return False


def run_ut():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    cov_dir = os.path.dirname(f"{cur_dir}/../../python/msprobe")
    report_dir = os.path.join(cur_dir, "report")
    cov_config_path = os.path.join(cur_dir, ".coveragerc")
    final_xml_path = os.path.join(report_dir, "final.xml")
    html_cov_report = os.path.join(report_dir, "htmlcov")
    xml_cov_report = os.path.join(report_dir, "coverage.xml")

    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    head_ut_directories = [os.path.join(cur_dir, "common_set_up")]
    tail_ut_directories = [os.path.join(cur_dir, "pytorch_ut"), os.path.join(cur_dir, "mindspore_ut")]
    msaccucmp_ut_dir = os.path.join(cur_dir, "msaccucmp_ut")

    # 分离 msaccucmp_ut 和其他 ut
    other_ut_dirs = []
    for item in os.listdir(cur_dir):
        item_path = os.path.join(cur_dir, item)
        if item == "msaccucmp_ut":
            continue
        if os.path.isdir(item_path) and not item.startswith("."):
            other_ut_dirs.append(item_path)
    other_ut_dirs.sort()
    
    # 处理其他 ut 的执行顺序
    for dir in head_ut_directories:
        if dir in other_ut_dirs:
            other_ut_dirs.remove(dir)
            other_ut_dirs.insert(0, dir)
    for dir in tail_ut_directories:
        if dir in other_ut_dirs:
            other_ut_dirs.remove(dir)
            other_ut_dirs.append(dir)

    # 检查 msaccucmp_ut 是否存在
    msaccucmp_ut_dirs = []
    if os.path.exists(msaccucmp_ut_dir) and os.path.isdir(msaccucmp_ut_dir):
        msaccucmp_ut_dirs = [msaccucmp_ut_dir]

    
    msaccucmp_ut_pythonpath = os.path.realpath(os.path.join(cur_dir, "../../python/msprobe/msaccucmp"))
    other_ut_pythonpath = os.path.realpath(os.path.join(cur_dir, "../../python"))
    
    all_success = True
    
    # 先执行 msaccucmp_ut
    if msaccucmp_ut_dirs:
        logger.info("Starting msaccucmp_ut execution with PYTHONPATH set...")
        success = _run_pytest(
            msaccucmp_ut_dirs,
            msaccucmp_ut_pythonpath,
            report_dir,
            cov_dir,
            cov_config_path,
            final_xml_path,
            html_cov_report,
            xml_cov_report,
            "msaccucmp_ut"
        )
        all_success = all_success and success
        
        # 清除 msaccucmp_ut 的 PYTHONPATH（为后续其他 ut 做准备）
        logger.info("Clearing msaccucmp_ut PYTHONPATH after execution...")
        if "MSACCUCMP_UT_PYTHONPATH" in os.environ:
            # 注意：这里只是日志说明，实际每个 subprocess 都有独立的环境变量
            # 不会影响后续执行
            logger.info("msaccucmp_ut PYTHONPATH cleared from environment")
    
    # 重新设置其他 ut 的 PYTHONPATH 并执行其他 ut
    if other_ut_dirs:
        logger.info("Setting other_ut PYTHONPATH and starting execution...")
        success = _run_pytest(
            other_ut_dirs,
            other_ut_pythonpath,
            report_dir,
            cov_dir,
            cov_config_path,
            final_xml_path,
            html_cov_report,
            xml_cov_report,
            "other_ut"
        )
        all_success = all_success and success
    
    if all_success:
        logger.info("All unit tests executed successfully.")
        return True
    else:
        logger.error("Some unit tests execution failed.")
        return False


if __name__ == "__main__":
    if run_ut():
        sys.exit(0)
    else:
        sys.exit(1)
