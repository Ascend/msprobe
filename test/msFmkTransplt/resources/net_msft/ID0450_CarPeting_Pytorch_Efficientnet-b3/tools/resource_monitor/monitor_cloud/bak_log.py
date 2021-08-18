#!/usr/bin/python3
# coding=gbk

import time
import os
import shutil
import ascend_function

class BakLog:
    def __init__(self):
        self.bak_log_path = "syslog"
        # self.log_list = ["device-0", "device-1", "device-2", "device-3", "device-4", "device-5", "device-6", "device-7",
        #                 "device-os-0", "device-os-1", "host-0"]

    def bak_log(self, global_dict):

        log_check_interval = global_dict.get("log_check_interval", 0)
        SYS_LOG_PATH = global_dict.get("SYS_LOG_PATH", "")

        # 创建存放系统日志的目录
        baseLogPath = os.path.join(os.getcwd(), self.bak_log_path)
        if os.path.exists(baseLogPath):
            filelist=os.listdir(baseLogPath)
            for file in filelist:
                filepath = os.path.join(baseLogPath, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                else:
                    pass
        else:
            os.makedirs(baseLogPath)


        # 设置初始归档Flag
        int_last_seconds = time.time()
        while True:
            int_current_seconds = time.time()

            # 备份错误日志
            cost_time = int_current_seconds - int_last_seconds
            if  cost_time > log_check_interval:

                # 搜集docker内的host日志
                container_path = os.path.join(SYS_LOG_PATH, "container/")
                strcmd = "find " + container_path + " -name \"*.log\" | xargs grep -l \"ERROR\""
                docker_result = os.popen(strcmd).read().strip('\n')
                docker_log_list = docker_result.split("\n")
                for log in docker_log_list:
                    if len(log) > 0:
                        basedir = os.path.dirname(log)
                        tmpdir = basedir.split(container_path)[-1]
                        dst_path = os.path.join(baseLogPath, tmpdir)
                        if not os.path.exists(dst_path):
                            os.makedirs(dst_path)
                        try:
                            shutil.copy(log, dst_path)
                        except:
                            continue

                # 搜集device的ERROR/WARNING日志
                dirlist = os.listdir(SYS_LOG_PATH)
                for item in dirlist:
                    if "csa" in item or item in ["container", "slogd"]:
                        continue
                    srcLogPath = os.path.join(SYS_LOG_PATH, item)
                    if os.path.exists(srcLogPath):
                        bakLogPath = os.path.join(baseLogPath, item)
                        if not os.path.exists(bakLogPath):
                            os.makedirs(bakLogPath)
                        strcmd = "egrep -l \"ERROR\" " + SYS_LOG_PATH + "/" + item + "/*.log"
                        result = os.popen(strcmd).read().strip('\n')
                        log_list = result.split("\n")
                        for log in log_list:
                            if len(log) > 0:
                                try:
                                    shutil.copy(log, bakLogPath)
                                except:
                                    continue

                # 更新上次备份时间
                int_last_seconds = int_current_seconds