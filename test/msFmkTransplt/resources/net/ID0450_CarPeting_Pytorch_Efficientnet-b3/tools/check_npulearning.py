# -*- coding:utf-8 -*-
'''
Created on 2020-06-08

@author: wwx371270
'''
import sys
import commands
import os
import csv


##########################################################################
####比较Bert_NZ中test_npu_output_tensor.txt与train_x.log的loss值是否一致####
##########################################################################

def checktensortxt(train_log_dir):
    flag = 0
    # 判断环境上有多少device
    device_list = commands.getoutput("ls /dev/|grep davinci|grep -v davinci_manager").split("\n")
    # 循环检查训练后的train_x.log
    for n in range(0, device_list.__len__()):
        print("check device" + str(n) + ":")
        # test_npu_output_tensor.txt
        tensortxt = "/autotest/CI_daily/Bert_NZ/code/" + str(n) + "/test_npu_output_tensor.txt"
        # train_x.log
        train_log = train_log_dir + "/train_" + str(n) + ".log"
        # 读取test_npu_output_tensor.txt每一行
        with open(tensortxt, 'r') as f:
            list_result = f.readlines()
        # 获取最后一行的step
        last_step = str(list_result[-1].split(",")[0].strip("step:"))
        # 把最后一行的结果保存在step_list列表中
        step_list = [list_result[-1]]
        if list_result.__len__() > 1:
            for n in range(0, list_result.__len__() - 1):
                # 从txt的倒数第二行开始向前循环
                n = -2 - n
                step = str(list_result[n].split(",")[0])
                # 如果最后一行的step还没第二次出现，说明还没遍历完最后一次的训练结果，如果出现了，则终止循环
                if last_step not in step:
                    # 遍历出来的结果保存在step_list列表中
                    step_list.append(list_result[n])
                else:
                    break
            # 循环step_list列表
            for n in range(0, step_list.__len__()):
                step = step_list[n].split(",")[0].strip("step:")
                step = "global_step = " + step + ","
                # 获取step_list里的loss值
                loss = step_list[n].strip("\n").split(",")[1].strip("loss:")
                # 获取train_x.log里的训练结果
                cmd = "cat " + str(train_log) + "| grep \"global_step =\""
                train_data_list = commands.getoutput(cmd).split("\n")
                for m in range(0, train_data_list.__len__()):
                    # 两边step相同时进行比较loss值
                    if step in train_data_list[m]:
                        train_loss = train_data_list[m].split(",")[-1].strip("total_loss = ").split(" ")[0]
                        if loss == train_loss:
                            print("step:" + str(step) + ", tensor_loss:" + loss + ", train_loss:" + train_loss + ", OK")
                        else:
                            print(
                                "step:" + str(step) + ", tensor_loss:" + loss + ", train_loss:" + train_loss + ", fail")
                            flag = 1
        # 当test_npu_output_tensor.txt只有一行的时候
        else:
            step = list_result[0].split(",")[0].strip("step:")
            step = "global_step = " + step + ","
            loss = list_result[0].strip("\n").split(",")[1].strip("loss:")
            cmd = "cat " + str(train_log) + "| grep \"global_step =\""
            train_data_list = commands.getoutput(cmd).split("\n")
            for m in range(0, train_data_list.__len__()):
                if step in train_data_list[m]:
                    train_loss = train_data_list[m].split(",")[-1].strip("total_loss = ").split(" ")[0]
                    if loss == train_loss:
                        print("step:" + str(step) + ", tensor_loss:" + loss + ", train_loss:" + train_loss + ", OK")
                    else:
                        print("step:" + str(step) + ", tensor_loss:" + loss + ", train_loss:" + train_loss + ", fail")
                        flag = 1
    if flag == 0:
        sys.exit(0)
    else:
        sys.exit(1)


def checktfprintexist(train_log_dir):
    flag = 0
    # 判断环境上有多少device
    device_list = commands.getoutput("ls /dev/|grep davinci|grep -v davinci_manager").split("\n")
    # 循环检查训练后的train_x.log
    for n in range(0, device_list.__len__()):
        print("check device" + str(n) + ":")
        # train_x.log
        train_log = train_log_dir + "/train_" + str(n) + ".log"
        # 获取train_x.log里的训练结果
        cmd = "cat " + str(train_log) + "| grep \"io.TextIOWrapper\""
        tfprint_list = commands.getoutput(cmd)
        if tfprint_list != "":
            print("[success]tf.print exists")
        else:
            print("[error]tf.print doesn't exist")
            flag = 1
    if flag == 0:
        sys.exit(0)
    else:
        sys.exit(1)


def checktfprintdata(train_log_dir):
    flag = 0
    # 判断环境上有多少device
    device_list = commands.getoutput("ls /dev/|grep davinci|grep -v davinci_manager").split("\n")
    # 循环检查训练后的train_x.log
    for n in range(0, device_list.__len__()):
        print("check device" + str(n) + ":")
        # train_x.log
        train_log = train_log_dir + "/train_" + str(n) + ".log"
        # 获取train_x.log里的训练结果
        cmd1 = "cat " + str(train_log) + "| grep \"io.TextIOWrapper\""
        tfprint_list = commands.getoutput(cmd1).split("\n")
        for n in range(0, tfprint_list.__len__()):
            tf_step = float(tfprint_list[n].split("<")[0].strip("[\"")) * 2 - 5
            tf_step = int(tf_step)
            if tf_step % 10 == 0:
                cmd2 = "cat " + str(train_log) + "| grep \"global_step = \"" + str(tf_step) + ","
                train_data = commands.getoutput(cmd2)
                train_loss = train_data.split(",")[-1].strip("total_loss = ").split(" ")[0]
                tf_loss = tfprint_list[n + 1].split("<")[0].strip("[\"")
                train_loss = round(float(train_loss) * 1000000) / 1000000
                tf_loss = round(float(tf_loss) * 1000000) / 1000000
                if train_loss == tf_loss:
                    print("step:" + str(tf_step) + ", train_loss:" + str(train_loss) + ", tf_loss:" + str(
                        tf_loss) + ", OK")
                else:
                    print("step:" + str(tf_step) + ", train_loss:" + str(train_loss) + ", tf_loss:" + str(
                        tf_loss) + ", fail")
                    flag = 1
    if flag == 0:
        sys.exit(0)
    else:
        sys.exit(1)


def checksummary():
    flag = 0
    # 判断环境上有多少device
    device_list = commands.getoutput("ls /dev/|grep davinci|grep -v davinci_manager").split("\n")
    for n in range(0, device_list.__len__()):
        print("check device" + str(n) + ":")
        file_dir = "/autotest/CI_daily/Bert_NZ/code/" + str(n) + "/summary_file/"
        cmd = "ls " + file_dir
        summary_exist = commands.getoutput(cmd)
        if "events.out.tfevents" in summary_exist:
            print("[success]summary file exists")
        else:
            print("[error]summary file doesn't exist")
            flag = 1
    if flag == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    train_log = sys.argv[1]
    if sys.argv[2] == "1":
        checktensortxt(train_log)
    if sys.argv[2] == "2":
        checksummary()
    if sys.argv[2] == "3":
        checktfprintexist(train_log)
    if sys.argv[2] == "4":
        checktfprintdata(train_log)

