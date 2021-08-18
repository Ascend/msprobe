# -*- encoding: utf-8 -*-
import sys
import os
import re
"""
函数说明：从训练打屏文件log_file中读取loss值到loss_array列表中
入参说明：log_file训练打屏文件
loss_array，loss值列表
"""
def read_loss(log_file):
    with open(log_file, 'r') as f:
        loss_lines = f.readlines()
        loss_array = []
        for line in loss_lines:
            if "end" not in line:
                result = re.findall(r'\d+\.*\d*', line)
                loss_array.append(float(result[0]))
    return loss_array

def cmp_loss(actual_loss, base_loss, num, value):
    err_loss_num = 0
    tmp_num = len(actual_loss)
    if tmp_num > int(num):
        tmp_num = int(num)
    for i in range(0, tmp_num):
        abs_loss = abs(float(actual_loss[i]) - float(base_loss[i]))
        if abs_loss > float(value):
            err_loss_num +=1
    print("err_loss_num:", err_loss_num)
    return err_loss_num


def main(argv):
    actual = argv[1]
    base = argv[2]
    num = argv[3]
    value = argv[4]
    actual_loss = read_loss(actual)
    base_loss = read_loss(base)
    cmp_loss(actual_loss, base_loss, num, value)




if __name__ == "__main__":
    main(sys.argv)