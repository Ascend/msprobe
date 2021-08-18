# -*- encoding: utf-8 -*-
import sys
import os
"""
函数说明：从训练打屏文件log_file中读取loss值到loss_array列表中
入参说明：log_file训练打屏文件
loss_array，loss值列表
"""
def read_loss_log(log_file):
    with open(log_file, 'r') as f:
        loss_lines = f.readlines()
        loss_array = []
        for line in loss_lines:
            if "reconstruction_loss" in line:
                temp_total_loss_data = line.split(': ')[1].strip('\n').strip('\"')
                loss_array.append(temp_total_loss_data)
    return loss_array

"""
函数说明：从训练打屏文件log_file中读取Iteration_Time值到time_array列表中
入参说明：log_file训练打屏文件
time_array，Iteration_Time值列表
"""
def read_time_log(log_file):
    with open(log_file, 'r') as f:
        loss_lines = f.readlines()
        time_array = []
        for line in loss_lines:
            if "Iteration_Time" in line:
                temp_total_loss_data = line.split(': ')[1].strip('\n').strip('\"')
                time_array.append(temp_total_loss_data)
    return time_array


"""
函数说明：从训练打屏文件log_file中读取precision值到precision_array列表中
入参说明：log_file训练打屏文件
precision_array，precision值列表
"""
def read_precision_log(log_file):
    with open(log_file, 'r') as f:
        precision_lines = f.readlines()
    precision_array = []
    for line in precision_lines:
        temp_precision_data = line.split(';')[1].split(':')[1]
        precision_array.append(temp_precision_data)
    return precision_array


"""
函数说明：从训练打屏文件log_file中读取fps值到fps_array列表中
入参说明：log_file训练打屏文件
fps_array，fps值列表
"""
def read_fps_log(log_file):
    with open(log_file, 'r') as f:
        fps_lines = f.readlines()
    fps_array = []
    for line in fps_lines:
        temp_total_fps_data = line.split(',')[2].split(':')[1]
        if "(" in temp_total_fps_data:
            total_fps_data = temp_total_fps_data.split("(")[0]
        else:
            total_fps_data = temp_total_fps_data
        fps_array.append(total_fps_data)
    return fps_array

"""
函数说明：实际loss和基准loss做比较
入参说明：实际loss--loss_array，基准loss-gt_loss，偏差值--abs_value
返回值说明：err_loss_num， loss检验失败的次数
"""
def cmp_loss(loss_array, gt_loss, abs_value):
    print('<===cmp_loss start===>')
    err_loss_num = 0
    start = 0
    end = len(gt_loss)
    loop = 1
    if type(gt_loss) is dict:
        keys = []
        for i in gt_loss:
            keys.append(i)
        print(keys)
        loop = int(keys[1]) - int(keys[0])
        start = int(keys[0])
        end = int(keys[-1])+1
    for step in range(start, end, loop):
        abs_loss = abs(float(loss_array[step]) - gt_loss[step])
        if abs_loss > abs_value[step]:
            err_loss_num += 1
            print("error_loss==> %r ,abs_loss==> %r ,expected_loss==> %r" % (loss_array[step], abs_value[step], gt_loss[step]))
        else:
            print("actual_loss==> %r ,abs_loss==> %r ,expected_loss==> %r" % (loss_array[step], abs_value[step], gt_loss[step]))
    print('<===cmp_loss end===>')
    return err_loss_num

"""
函数说明：实际iteration_time和基准iteration_time做比较
入参说明：实际time--time_array，基准time--gt_time，偏差值--abs_value
返回值说明：err_loss_num， iteration_time检验失败的次数
"""
def cmp_time(time_array, gt_time, abs_value):
    print('<===cmp_iteration_time start===>')
    err_loss_num = 0
    start = 0
    end = len(gt_time)
    loop = 1
    if type(gt_time) is dict:
        keys = []
        for i in gt_time:
            keys.append(i)
        print(keys)
        loop = int(keys[1]) - int(keys[0])
        start = int(keys[0])
        end = int(keys[-1])+1
    for step in range(start, end, loop):
        abs_loss = abs(float(time_array[step]) - gt_time[step])
        if abs_loss > abs_value[step]:
            err_loss_num += 1
            print("error_time==> %r ,abs_time==> %r ,expected_time==> %r" % (time_array[step], abs_value[step], gt_time[step]))
        else:
            print("actual_time==> %r ,abs_time==> %r ,expected_time==> %r" % (time_array[step], abs_value[step], gt_time[step]))
    print('<===cmp_iteration_time end===>')
    return err_loss_num

"""
函数说明：实际precision和基准precision做比较
入参说明：实际precision--precision_array，基准precision--gt_precision
返回值说明：err_precision_num， precision检验失败的次数
"""
def cmp_precision(precision_array, gt_precision):
    err_precision_num = 0
    for step in range(1, 10, 2):
        abs_precision = abs(float(precision_array[step]) - gt_precision[step])
        if abs_precision > 0.5:
            err_precision_num += 1
            print("err_precision==> %r ,gt_precision==> %r" % (precision_array[step], gt_precision[step]))
        else:
            print("precision==> %r ,gt_precision==> %r" % (precision_array[step], gt_precision[step]))
    return err_precision_num

"""
函数说明：实际fps和基准fps做比较
入参说明：实际fps--perf_array，基准fps--gt_perf，偏差值--abs_value
返回值说明：err_perf_num， fps检验失败的次数
"""
def cmp_fps(perf_array, gt_perf, abs_value):
    print('<===cmp_fps start===>')
    err_perf_num = 0
    start = 0
    end = len(gt_perf)
    loop = 1
    if type(gt_perf) is dict:
        keys = []
        for i in gt_perf:
            keys.append(i)
        print(keys)
        loop = int(keys[1]) - int(keys[0])
        start = int(keys[0])
        end = int(keys[-1])+1
    for step in range(start, end, loop):
        abs_perf = abs(float(perf_array[step]) - gt_perf[step])
        if float(perf_array[step]) < gt_perf[step]:
            err_perf_num += 1
            print("error_fps==> %r ,abs_fps==> %r ,expected_fps==> %r" % (perf_array[step], abs_value[step], gt_perf[step]))
        else:
            print("actual_fps==> %r ,abs_fps==> %r ,expected_fps=> %r" % (perf_array[step], abs_value[step], gt_perf[step]))
    print('<===cmp_fps end===>')
    return err_perf_num

"""
函数说明：实际值和基准值做比较
入参说明：比较内容用于信息打屏--content，字符串格式，实际值--loss_array，基准值--gt_loss，偏差值--abs_value
支持list、tuple、dict格式，但入参格式需保持一致
返回值说明：err_loss_num， 检验失败的次数
"""
def cmp(content, loss_array, gt_loss, abs_value):
    print('<===cmp', content, 'start===>')
    err_loss_num = 0
    start = 0
    end = len(gt_loss)
    loop = 1
    if type(gt_loss) is dict:
        keys = []
        for i in gt_loss:
            keys.append(i)
        print(keys)
        loop = int(keys[1]) - int(keys[0])
        start = int(keys[0])
        end = int(keys[-1])+1
    for step in range(start, end, loop):
        abs_loss = abs(float(loss_array[step]) - gt_loss[step])
        if abs_loss > abs_value[step]:
            err_loss_num += 1
            print("error_%r==> %r ,abs_%r==> %r ,expected_%r==> %r" % (content, loss_array[step], content, abs_value[step], content, gt_loss[step]))
        else:
            print("actual_%r==> %r ,abs_%r==> %r ,expected_%r==> %r" % (content, loss_array[step], content, abs_value[step], content, gt_loss[step]))
    print('<===cmp', content, 'end===>')
    return err_loss_num

"""
函数说明：获取环境型态和cpu核数信息
入参说明：无
返回值说明：host_type硬件型态arm或x64，cpu核数
"""
def get_envinfo():
    result1 = os.popen('uname -m').readlines()
    result2 = os.popen("cat /proc/cpuinfo | grep 'process' | sort | uniq | wc -l").readlines()
    host_type = result1[0]
    cpu_num = result2[0]
    print(host_type)
    print(cpu_num)
    return host_type, cpu_num


def main(argv):
    if len(argv) == 1:
        rank_size = 1
    else:
        rank_size = argv[1]
    print(rank_size)
    get_envinfo()
    # loss_array = read_loss_log("/data/Unet2D/Results/collect.log")
    # time_array = read_time_log("/data/autotest/Unet2D/Results/collect.log")
    if int(rank_size) == 1:

        loss_array = {10: 1, 20: 0.5, 40: 0.175, 60: 0.16}
        time_array = (0.45, 0.45, 0.45)
        gt_loss = {20: 0.6, 40: 0.175, 60: 0.16}
        abs_loss = {20: 0.45, 40: 0.1, 60: 0.1}
        gt_time = (0.45, 0.45, 0.45)
        abs_time = (0.03, 0.03, 0.03)
    else:
        loss_array = {1: 1, 2: 0.5, 4: 0.175, 6: 0.16}
        time_array = (0.45, 0.45, 0.45)
        gt_loss = {2: 0.6, 4: 0.175, 6: 0.16}
        abs_loss = {2: 0.45, 4: 0.1, 6: 0.1}
        gt_time = (0.45, 0.45, 0.45)
        abs_time = (0.03, 0.03, 0.03)
    #err_loss_num = cmp("loss", loss_array, gt_loss, abs_loss)
    err_loss_num = cmp_loss(loss_array, gt_loss, abs_loss)
    #err_loss_time = cmp("time", time_array, gt_time, abs_time)
    err_loss_time = cmp_time(time_array, gt_time, abs_time)
    if err_loss_num > 0 or err_loss_time >0:
       print("error")
    assert err_loss_num < 1, "error loss"
    assert err_loss_time < 1, "error time"


if __name__ == "__main__":
    main(sys.argv)
