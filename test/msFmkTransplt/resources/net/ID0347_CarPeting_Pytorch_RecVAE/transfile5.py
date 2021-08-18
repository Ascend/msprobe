import re
import time
import os
import argparse
import sys


def func(trans_dir):
    sign = "[TASK_INFO]"
    res_list = []
    pid_list = []
    trans_dir = os.path.join(os.getcwd(), trans_dir)
    os.chdir(trans_dir)
    trans_list = [i for i in os.listdir() if i.endswith(".log")]
    trans_list.sort()
    print(f"input:{trans_list}")
    out_data = ""
    for j in trans_list:
        with open(j, encoding="utf-8", errors="ignore") as f:
        #with open(j, mode="r", encoding="utf-8") as f:
            for res in f:
                if sign in res:
                    if res.strip().endswith("tvmbin"):
                        m = re.search(r'\[TASK_INFO\] (\d*)\/.*tvmbin', res.strip())
                        n = re.search(r'\((?P<pid>\d*),python3\)', res)
                        if n.group('pid') not in pid_list:
                            pid_list.append(n.group('pid'))
                        for i in res_list:
                            if m.group(0) in i:
                                res = res.replace(res, i)
                                reslist = res.split(sign)
                                res1 = reslist[1]
                                out_data += (f"[{n.group('pid')}]" + res1)
                        continue
                    res_list.append(res)
    print(pid_list)
    for k in pid_list:
        pid_data = ''
        filename = f"task_{k}.txt"
        with open(filename, mode="w") as f:
            f.write(out_data)
        with open(filename, mode="r") as f:
            for p in f:
                if k in p:
                    res = len(k)
                    pid_data += p[res+3 :]
        with open(filename, mode="w") as f:
            f.write(pid_data)
        file_dir = os.path.join(sys.path[0], filename)
        os.system(f"mv {filename} {file_dir}")
        print(f"output:{filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trans the log')
    parser.add_argument('--dir', default="",
                        help="input the dir name,trans the current dir with default")
    ags = parser.parse_args()
    func(ags.dir)
