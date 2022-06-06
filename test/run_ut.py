import os
import subprocess
import shutil
import sys


def run_ut():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    top_dir = os.path.abspath(os.path.dirname(cur_dir))
    ut_path = os.path.join(cur_dir, "ut/testcase/")
    src_dir = os.path.join(top_dir, "src/compare")
    report_dir = os.path.join(cur_dir, 'report')

    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)

    os.makedirs(report_dir)

    cmd = ['python3', '-m', 'pytest', ut_path, '--junitxml=' + report_dir + '/final.xml',
           '--cov=' + src_dir, '--cov-branch', '--cov-report=xml:' + report_dir + '/coverage.xml']

    result_ut = subprocess.Popen(cmd, shell=False,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)

    while result_ut.poll() is None:
        line = result_ut.stdout.readline()
        line = line.strip()
        if line:
            print(line)

    ut_flag = False
    if result_ut.returncode == 0:
        ut_flag = True
        print("run ut success")
    else:
        print("run ut failed")

    return ut_flag


if __name__ == "__main__":
    if run_ut():
        sys.exit(0)
    else:
        sys.exit(-1)
