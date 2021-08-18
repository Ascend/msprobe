import multiprocessing
import os
import sys
import threading
import time
import subprocess
from ctypes import *
import ascend_function


def excute_dsmi(so_route):
    libDsmi = cdll.LoadLibrary(so_route)
    function = libDsmi.func_dsmi_get_device_health
    ff = function
    ff.argtypes = [c_int]
    ff.restype = c_char_p
    while 1:
        function(0, 250000)


def dsmi_thread(so_route):
    # 线程池
    threads = []
    for i in range(0, 16):
        th = threading.Thread(target=excute_dsmi, args=(so_route,))
        th.start()
        threads.append(th)
    #     # time.sleep(1)
    # # 等待线程运行完毕
    # for th in threads:
    #     th.join()


def dsmi_process(so_route):
    process_list = []
    for i in range(0, 17):
        pc = multiprocessing.Process(target=excute_dsmi, args=(so_route,))
        process_list.append(pc)
        pc.start()


if __name__ == "__main__":
    route = subprocess.getoutput("pwd")
    sys = subprocess.getoutput("uname -r")
    if "aarch64" in sys:
        so_route = route + "/turing_dsmi_monitor_arm.so"
    elif "x86" in sys:
        so_route = route + "/turing_dsmi_monitor_x86.so"
    print(so_route)
    # dsmi_thread(so_route)
    dsmi_process(so_route)

