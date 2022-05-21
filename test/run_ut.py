import os
import subprocess


def run_ut():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    ut_path = os.path.join(cur_dir, "ut/testcase/")
    src_name = "compare"

    cmd = ['python3.9', '-m', 'pytest', ut_path,
           '--cov=' + src_name, '--cov-report=xml:'
           + os.path.join(cur_dir, 'ut_report.xml')]

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
    run_ut()
