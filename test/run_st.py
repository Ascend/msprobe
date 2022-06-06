import os
import subprocess


def run_st():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    top_dir = os.path.abspath(os.path.dirname(cur_dir))
    st_path = os.path.join(cur_dir, "st/testcase/")
    src_dir = os.path.join(top_dir, "src/compare")

    cmd = ['python3', '-m', 'pytest', st_path,
           '--cov=' + src_dir, '--cov-report=xml:'
           + os.path.join(cur_dir, 'st_report.xml')]

    result_st = subprocess.Popen(cmd, shell=False,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)

    while result_st.poll() is None:
        line = result_st.stdout.readline()
        line = line.strip()
        if line:
            print(line)

    st_flag = False
    if result_st.returncode == 0:
        st_flag = True
        print("run st success")
    else:
        print("run st failed")

    return st_flag


if __name__ == "__main__":
    run_st()
