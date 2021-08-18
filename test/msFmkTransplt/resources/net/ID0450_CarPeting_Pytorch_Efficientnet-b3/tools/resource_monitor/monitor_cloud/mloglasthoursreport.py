#!/usr/bin/python3
# coding=gbk
import os
import sys
import time
import datetime

if __name__ == "__main__":
    int_hours = 1
    if len(sys.argv) > 1:
        int_hours = int(sys.argv[1])
    int_current_time = datetime.datetime.now()
    str_end_time = int_current_time.strftime("%Y-%m-%d %H:%M:%S")

    int_next_time = int_current_time - datetime.timedelta(hours=int_hours)
    str_begin_time = int_next_time.strftime("%Y-%m-%d %H:%M:%S")

    print(os.popen('./mlogreport.py "" ' + '"'+ str_begin_time +'" "' + str_end_time + '"').read())
