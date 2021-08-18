# -*- coding:utf-8 -*-
import commands
import time
import datetime
import sys
import os
import json
import ascend_function

pwd = commands.getoutput("pwd")
with open(pwd + "/config.json", 'r') as load_f:
    load_dict = json.load(load_f)


# 版本
def get_version():
    # cmd = "cat /usr/local/Ascend/driver/version.info | grep Version"
    # version = commands.getoutput(cmd).replace("Version=", "")
    return load_dict["version"]


# 包路径
def get_pkgroute():
    return load_dict["pkg_route"]


# 开始时间
def get_starttime(case):
    caseroute = commands.getoutput(
        "find /autotest/CI_daily/ -name " + str(case) + " | grep -v training_shop")
    route = caseroute.replace("testscript/" + str(case), "") + "result/"
    cmd = "ls -lrt " + str(route) + " | grep cloud | head -n 1 |awk '{print $9}'"
    cloud = commands.getoutput(cmd)
    cloud_route = str(route) + str(cloud)
    train_file = cloud_route + "/train.sh"
    starttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(train_file).st_mtime))
    return str(starttime)


# 模块
def get_module():
    return "Integration"


# 用例名
def get_casename(case):
    return str(case).replace(".sh", "")


# 芯片
def get_chipset():
    return "1980"


# 硬件形态
def get_hardware():
    sys = commands.getoutput("uname -r")
    if "aarch64" in sys:
        return "AIServer_A+K"
    elif "x86" in sys:
        return "AIServer_A+X"


# 软件形态
def get_software(case):
    caseroute = commands.getoutput(
        "find /autotest/CI_daily/ -name " + str(case) + " | grep -v training_shop")
    route = caseroute.replace("testscript/" + str(case), "") + "result/"
    cmd1 = "ls -lrt " + str(route) + "| grep cloud | head -n 1 |awk '{print $9}'"
    cloud_route = str(route) + str(commands.getoutput(cmd1))
    cmd2 = "grep \"export exec_type\" " + cloud_route + "/cloud_docker_init.sh"
    type = commands.getoutput(cmd2)
    if "docker" in type:
        return "docker-os"
    elif "host" in type:
        return "host-os"
    elif "vm" in type:
        return "vm-os"


# 组网
def get_networking(case):
    case = str(case)
    case = case.replace(".sh", "")
    case_list = case.split("_")
    for n in range(0, case_list.__len__()):
        if str(case_list[n][-1]) == "p":
            return str(case_list[n])


# ip
def get_ip():
    cmd = "ifconfig -a|grep inet|grep -v inet6|grep -v 192.168|awk '{print $2}'|tr -d addr: | head -n 2 | tail -n 1"
    ip = commands.getoutput(cmd)
    return str(ip)


# 执行时长
def get_excutetime(case):
    caseroute = commands.getoutput("find /autotest/CI_daily/ -name " + str(case) + " | grep -v training_shop")
    route = caseroute.replace("testscript/" + str(case), "") + "result/"
    cmd = "ls -lnt " + str(route) + " | grep cloud | head -n 1 |awk '{print $9}'"
    cloud = commands.getoutput(cmd)
    cloud_route = str(route) + str(cloud)
    train_file = cloud_route + "/train.sh"
    newesttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(train_file).st_mtime))
    newesttime = datetime.datetime.strptime(newesttime, '%Y-%m-%d %H:%M:%S')
    delta = newesttime - datetime.datetime.strptime(get_starttime(case), '%Y-%m-%d %H:%M:%S')
    hour = int(delta.total_seconds()) / 3600
    return str(hour) + "h"


# 执行次数
def get_excutenum(case):
    caseroute = commands.getoutput("find /autotest/CI_daily/ -name " + str(case) + " | grep -v training_shop")
    route = caseroute.replace("testscript/" + str(case), "") + "result/"
    cmd = "ls -lrt " + str(route) + " | grep cloud | awk '{print $9}'"
    cloud_route = commands.getoutput(cmd).split("\n")
    num = cloud_route.__len__()
    return str(num)


# 日志，黑匣子，coredump
def get_log():
    cmd1 = "ls /autotest/monitor_cloud/syslog/"
    cmd2 = "ls /var/log/hisi_logs"
    cmd3 = "ls /var/log/npu/dump | grep core"
    cmd4 = "ls /var/log/npu/dump/container/cloud-localhost* | grep core"
    flag = 0
    slog = commands.getoutput(cmd1)
    if slog != "" and "No such file or directory" not in slog:
        flag = 1
    hisilog = commands.getoutput(cmd2)
    if hisilog != "" and "No such file or directory" not in hisilog:
        flag = 1
    core1 = commands.getoutput(cmd3)
    if core1 != "" and "No such file or directory" not in core1:
        flag = 1
    core2 = commands.getoutput(cmd4)
    if core2 != "" and "No such file or directory" not in core2:
        flag = 1
    if flag == 0:
        return "0"
    else:
        return "1"


# 计算性能并比较
def caculate_fps(fps_list, net):
    fps_total = 0
    for n in range(0, 100):
        fps_total += float(fps_list[n - 100])
        # print(fps_list[n - 10])
    fps_ava = fps_total / 100
    if fps_ava > load_dict[net]:
        return "0"
    else:
        return "1"


# 性能
def get_fps(case):
    caseroute = commands.getoutput("find /autotest/CI_daily/ -name " + str(case) + " | grep -v training_shop")
    route = caseroute.replace("testscript/" + str(case), "") + "result/"
    cmd = "ls -lnt " + str(route) + "| grep cloud | head -n 1 |awk '{print $9}'"
    trainlog = str(route) + str(commands.getoutput(cmd)) + "/train_0.log"
    trainresult = commands.getoutput("grep \"turing train fail\" " + trainlog)
    fps_list = []
    if os.path.exists(trainlog) == False or trainresult != "":
        return "1"
    else:
        if "resnet50_TF" in case:
            fps_grep = commands.getoutput("grep \"FPS:\" " + trainlog)
            if "No such file or directory" not in fps_grep:
                if fps_grep != "":
                    fps_grep = fps_grep.split("\n")
                    if fps_grep.__len__() > 100:
                        for n in range(0, fps_grep.__len__()):
                            fps = fps_grep[n].split(",")
                            fps_list.append(fps[-3].replace("FPS:", ""))
                        return (caculate_fps(fps_list, "resnet50_TF"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"
        elif "resnet50_hc" in case:
            fps_grep = commands.getoutput("grep \"FPS:\" " + trainlog)
            if "No such file or directory" not in fps_grep:
                if fps_grep != "":
                    fps_grep = fps_grep.split("\n")
                    if fps_grep.__len__() > 100:
                        for n in range(0, fps_grep.__len__()):
                            fps = fps_grep[n].split(",")
                            fps_list.append(fps[-3].replace("FPS:", ""))
                        return (caculate_fps(fps_list, "resnet50_hc"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"
        elif "resnet50_nv" in case:
            fps_grep = commands.getoutput("grep \"imgs_per_sec:\" " + trainlog)
            if "No such file or directory" not in fps_grep:
                if fps_grep != "":
                    fps_grep = fps_grep.split("\n")
                    if fps_grep.__len__() > 100:
                        for n in range(0, fps_grep.__len__()):
                            fps = fps_grep[n].split(" ")[-1]
                            fps_list.append(fps)
                        return (caculate_fps(fps_list, "resnet50_nv"))
                    else:
                        return "1"
                else:
                    return "0"
            else:
                return "0"
        elif "vgg16_tf" in case:
            fps_grep = commands.getoutput(
                "grep \"images/sec:\" " + trainlog + " | grep -v total | awk -F ' ' '{print $3}'")
            if "No such file or directory" not in fps_grep:
                if fps_grep != "":
                    fps_list = fps_grep.split("\n")
                    if fps_grep.__len__() > 100:
                        return (caculate_fps(fps_list, "vgg16_tf"))
                    else:
                        return "0"
                else:
                    return "0"

            else:
                return "1"
        elif "alexnet_tf" in case:
            fps_grep = commands.getoutput(
                "grep \"images/sec:\" " + trainlog + " | grep -v total | awk -F ' ' '{print $3}'")
            if "No such file or directory" not in fps_grep:
                if fps_grep != "":
                    fps_list = fps_grep.split("\n")
                    if fps_grep.__len__() > 100:
                        return (caculate_fps(fps_list, "alexnet_tf"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"
        elif "bert_nv" in case:
            fps_grep = commands.getoutput("grep \"Throughput =\" " + trainlog + " | awk -F ' ' '{print $6}'")
            if "No such file or directory" not in fps_grep:
                if fps_grep != "":
                    fps_list = fps_grep.split("\n")
                    if fps_grep.__len__() > 100:
                        return (caculate_fps(fps_list, "bert_nv"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"


# 计算精度并比较
def caculate_acc(acc_list, net):
    acc_total_first = 0
    acc_total_last = 0
    for n in range(0, 100):
        acc_total_first += float(acc_list[n])
        acc_total_last += float(acc_list[n - 100])
    acc_ava_first = acc_total_first / 100
    acc_ava_last = acc_total_last / 100
    if acc_ava_first > acc_ava_last:
        return "0"
    else:
        return "1"


# 精度
def get_accuracy(case):
    caseroute = commands.getoutput(
        "find /autotest/CI_daily/ -name " + str(case) + " | grep -v training_shop")
    route = caseroute.replace("testscript/" + str(case), "") + "result/"
    cmd = "ls -lnt " + str(route) + "| grep cloud | head -n 1 |awk '{print $9}'"
    trainlog = str(route) + str(commands.getoutput(cmd)) + "/train_0.log"
    trainresult = commands.getoutput("grep \"turing train fail\" " + trainlog)
    acc_list = []
    if os.path.exists(trainlog) == False or trainresult != "":
        return "1"
    else:
        if "resnet50_TF" in case:
            acc_grep = commands.getoutput("grep \"loss =\" " + trainlog + " | grep global_step | awk '{print $18}'")
            if "No such file or directory" not in acc_grep:
                if acc_grep != "":
                    acc_list = acc_grep.split("\n")
                    print(acc_list)
                    if acc_grep.__len__() > 10:
                        return (caculate_acc(acc_list, "resnet50_TF"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"
        elif "resnet50_nv" in case:
            acc_grep = commands.getoutput(
                "grep total_loss: " + trainlog + " | grep -v final_total_loss: | awk -F ' ' '{print $5}'")
            if "No such file or directory" not in acc_grep:
                if acc_grep != "":
                    acc_list = acc_grep.split("\n")
                    if acc_grep.__len__() > 10:
                        return (caculate_acc(acc_list, "resnet50_nv"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"
        elif "vgg16_tf" in case:
            acc_grep = commands.getoutput(
                "grep \"images/sec:\" " + trainlog + " | grep -v 2020- | grep -v total | awk -F ' ' '{print $9}'")
            if "No such file or directory" not in acc_grep:
                if acc_grep != "":
                    acc_list = acc_grep.split("\n")
                    if acc_grep.__len__() > 10:
                        return (caculate_acc(acc_list, "vgg16_tf"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"
        elif "alexnet_tf" in case:
            acc_grep = commands.getoutput(
                "grep \"images/sec:\" " + trainlog + " | grep -v 2020- | grep -v total | awk -F ' ' '{print $9}'")
            if "No such file or directory" not in acc_grep:
                if acc_grep != "":
                    acc_list = acc_grep.split("\n")
                    if acc_grep.__len__() > 10:
                        return (caculate_acc(acc_list, "vgg16_tf"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"
        elif "bert_nv" in case:
            acc_grep = commands.getoutput("grep tensorflow:loss " + trainlog + " | awk -F ' ' '{print $3}'")
            if "No such file or directory" not in acc_grep:
                if acc_grep != "":
                    acc_grep = acc_grep.split("\n")
                    if acc_grep.__len__() > 10:
                        for n in range(0, acc_grep.__len__()):
                            acc_list.append(acc_grep[n].replace(",", ""))
                        return (caculate_acc(acc_list, "bert_nv"))
                    else:
                        return "0"
                else:
                    return "0"
            else:
                return "1"


def excute_monitor(case):
    nowtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    starttime = get_starttime(case)
    route = "/autotest/monitor_cloud"
    cmd = "cd " + route + " && ./mlogreport.py \"\" " + "\"" + starttime + "\" \"" + nowtime + "\""
    commands.getoutput(cmd)
    # print(cmd)


# 内存
def get_mem(case):
    cmd1 = "ls -lrt /autotest/monitor_cloud/hlog | awk '{print $9}'"
    hlog_list = commands.getoutput(cmd1).split("\n")
    hlog_first_route = "/autotest/monitor_cloud/hlog/" + str(hlog_list[1]) + "/"
    cmd2 = "ls -lrt " + hlog_first_route + " | awk '{print $9}'"
    hlog_first_list = commands.getoutput(cmd2).split("\n")
    hlog_first = hlog_first_route + str(hlog_first_list[1])

    mem_cmd1 = "zcat " + hlog_first + " | grep MemUsage"
    mem_first = commands.getoutput(mem_cmd1).split("\n")
    mem_first_total = 0
    for n in range(0, mem_first.__len__()):
        mem_first[n] = mem_first[n].replace("MemUsage(%):", "")
    for m in range(0, mem_first.__len__()):
        mem_first_total += float(mem_first[m])
    mem_first_ava = mem_first_total / mem_first.__len__()
    # print(mem_cmd1)
    # print(mem_first_ava)

    if "monitor.runlog" in str(hlog_list[-1]):
        hlog_last = "/autotest/monitor_cloud/hlog/monitor.runlog"
        mem_cmd2 = "cat " + hlog_last + " | grep MemUsage"
    else:
        hlog_last_route = "/autotest/monitor_cloud/hlog/" + str(hlog_list[-1]) + "/"
        cmd3 = "ls -lrt" + hlog_last_route + " | awk '{print $9}'"
        hlog_last_list = commands.getoutput(cmd3).split("\n")
        hlog_last = hlog_last_route + str(hlog_last_list[-1])
        mem_cmd2 = "zcat " + hlog_last + " | grep MemUsage"

    mem_last = commands.getoutput(mem_cmd2).split("\n")
    mem_last_total = 0
    for n in range(0, mem_last.__len__()):
        mem_last[n] = mem_last[n].replace("MemUsage(%):", "")
    for m in range(0, mem_last.__len__()):
        mem_last_total += float(mem_last[m])
    mem_last_ava = mem_last_total / mem_last.__len__()
    # print(mem_cmd2)
    # print(mem_last_ava)

    excute_monitor(case)
    flag = 0
    for n in range(0, 8):
        deviceroute = "/autotest/monitor_cloud/report_device_dlog" + str(n)
        txt_cmd = "ls -lrt " + deviceroute + " | awk '{print $9}' | grep txt"
        txtnew = commands.getoutput(txt_cmd).split("\n")[-1]
        txtnew = deviceroute + "/" + txtnew
        ddr_cmd = "cat " + txtnew + " | grep DDRUtiliza"
        hbm_cmd = "cat " + txtnew + " | grep HBMUtiliza"
        ddr = commands.getoutput(ddr_cmd).split(":")[1].split(",")
        hbm = commands.getoutput(hbm_cmd).split(":")[1].split(",")
        ddr_ava = float(ddr[0])
        ddr_max = float(ddr[1])
        hbm_ava = float(hbm[0])
        hbm_max = float(hbm[1])
        # print(ddr_ava, ddr_max)
        # print(hbm_ava, hbm_max)
        if ddr_ava / ddr_max > 0.7 and hbm_ava / hbm_max > 0.7:
            flag = 0
        else:
            flag = 1

    if mem_first_ava / mem_last_ava > 0.7 and flag == 0:
        return "0"
    else:
        return "1"


def get_reportroute():
    return "/autotest/monitor_cloud/report_*"


if __name__ == "__main__":
    case = load_dict["case"]

    version = get_version()
    print("version: " + version)
    pkgroute = get_pkgroute()
    print("pkgroute: " + pkgroute)
    starttime = get_starttime(case)
    starttime = str(starttime.split(" ")[0])
    print("starttime: " + starttime)
    module = get_module()
    print("module: " + module)
    casename = get_casename(case)
    print("casename: " + casename)
    chipset = get_chipset()
    print("chipset: " + chipset)
    hardware = get_hardware()
    print("hardware: " + hardware)
    software = get_software(case)
    print("software: " + software)
    networking = get_networking(case)
    print("networking: " + networking)
    ip = get_ip()
    print("ip: " + ip)
    excutetime = get_excutetime(case)
    print("excutetime: " + excutetime)
    excutenum = get_excutenum(case)
    print("excutenum: " + excutenum)
    log = get_log()
    print("log: " + log)
    fps = get_fps(case)
    print("fps: " + fps)
    accuracy = get_accuracy(case)
    print("accuracy:" + accuracy)
    mem = get_mem(case)
    print("mem: " + mem)
    reportroute = get_reportroute()
    print("reportroute: " + reportroute)

    txt_board = version + "|" + pkgroute + "|" + starttime + "|" + module + "|" + casename + "|" + chipset + "|" + hardware + "|" + software + "|" + \
                networking + "|" + ip + "|" + excutetime + "|" + excutenum + "|" + log + "|" + fps + "|" + accuracy + "|" + mem + "|" + reportroute

    fo = open("log_check.txt", "w")
    fo.write(txt_board)
    fo.close()
