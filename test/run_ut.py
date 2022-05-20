import os
import subprocess


def run_ut():
    ut_path = os.path.abspath("ut/testcase/")
    src_path = os.path.abspath("compare/")
    src_name = "compare"
    cur_dir = os.path.abspath('.')

    cmd = ['python3.7', '-m', 'pytest', ut_path,
           '--cov=' + src_name, '--cov-report=html:' + cur_dir + '/ut_report']
    print(cmd)

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
