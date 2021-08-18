#!/usr/bin/python3.7
# coding=utf-8

import time
import socket
import os
import struct
import fcntl
import ascend_function

class MonitorHost:
    def __init__(self):
        self.time_header = "OutputTime:"
        self.hostname_header = "HostName:"
        self.ip_header = "HostIPPort:"
        self.free_hugepages_header = "FreeHugePages:"
        self.free_mem_header = "FreeMem(MB):"
        self.mem_usage_percent_header = "MemUsage(%):"
        self.cpu_usage_header = "CpuUsage(%):"

        self.log_folder_name_h = "hlog"
        self.result_file_name_first = 'monitor'
        self.result_file_name_postfix = '.runlog'
        self.archive_result_file_name_postfix = '.gz'

        self.hostname = self.hostname_header + socket.gethostname()


    def get_filesize(self, filePath):
        fsize = os.path.getsize(filePath)
        return fsize


    def get_ip(self, ifname):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', bytes(ifname[:15],'utf-8')))[20:24])


    def monitor_host(self, global_dict):

        check_interval = global_dict.get("check_interval", 0)
        archive_interval = global_dict.get("archive_interval", 0)
        int_max_size_per_runlog = global_dict.get("int_max_size_per_runlog", 0)
        int_max_archive_day_num = global_dict.get("int_max_archive_day_num", 0)
        NETWORK_CARD = global_dict.get("NETWORK_CARD", "")
        HOST_PROCESS = global_dict.get("HOST_PROCESS", "")

        # init env
        result_file_name = self.result_file_name_first + self.result_file_name_postfix
        print(os.popen('mkdir -pv ' + self.log_folder_name_h).read())
        print(os.popen('rm -rf ' + self.log_folder_name_h + "/*").read())
        result_file_path_h = self.log_folder_name_h + "/" + result_file_name
        print(os.popen('touch ' + result_file_path_h).read())

        # 设置初始归档Flag
        int_last_seconds = time.time()
        g_archive_flag = int(int_last_seconds / archive_interval)
        st_last_time = time.localtime(int_last_seconds)
        while True:
            int_current_seconds = time.time()
            st_current_time = time.localtime(int_current_seconds)

            #判断是否需要打包压缩归档
            bool_need_archive_h = (st_last_time.tm_mday != st_current_time.tm_mday)

            archive_flag = int(int_current_seconds / archive_interval)
            if g_archive_flag:
                if g_archive_flag != archive_flag:
                    bool_need_archive_h = True


            # 判断host侧的日志是否超过日志文件最大size
            if (self.get_filesize(result_file_path_h)) > int_max_size_per_runlog:
                bool_need_archive_h = True

            if bool_need_archive_h:
                # 按照年月日创建目录
                str_archive_folder_name = self.log_folder_name_h + "/" + time.strftime("%Y%m%d", st_last_time)
                str_archive_file_name = self.result_file_name_first + '_' + time.strftime("%Y%m%d%H%M%S", st_last_time) + self.result_file_name_postfix
                print(os.popen('mkdir -pv ' + str_archive_folder_name).read())
                print(os.popen('mv ' + result_file_path_h + ' ' + str_archive_folder_name + '/' + str_archive_file_name + ';gzip -f ' + str_archive_folder_name + '/' + str_archive_file_name).read())

                # 超过90天的目录删除
                date_folder_names = os.listdir(self.log_folder_name_h)
                date_folder_names.sort()
                while len(date_folder_names) > int_max_archive_day_num:
                    old_folder = date_folder_names.pop(0)
                    print(os.popen('rm -rfv ' + self.log_folder_name_h + '/' + old_folder).read())

            # 重新赋值归档标识
            g_archive_flag = archive_flag
            st_last_time = st_current_time

            str_current_time = self.time_header + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int_current_seconds))

            # host侧搜集mem命令
            result = os.popen("cat /proc/meminfo | grep HugePages_Free | awk '{print $2}'").read()
            # print(result)
            str_free_hugepages = self.free_hugepages_header + result.strip('\n')

            result = os.popen("/usr/bin/free -m | awk 'NR==2' | awk '{print $4+$6}'").read()
            # print(result)
            str_free_mem = self.free_mem_header + result.strip('\n')

            result = os.popen("/usr/bin/free -m | awk '{if(NR==2){total=$2;used=$3}} END{printf(\"%.02f\", used*100/total)}'").read()
            # print(result)
            str_mem_usage_percent = self.mem_usage_percent_header + result.strip('\n')

            # host侧执行搜集CPU命令
            result = os.popen("/usr/bin/mpstat -P ALL 1 1 | grep \" all \" | grep -v \"Average:\" | awk '{print 100-$13}'").read()
            str_cpu_usage = self.cpu_usage_header + result.strip('\n')

            str_host_ip = self.ip_header + str(self.get_ip(NETWORK_CARD)) + '_22118'

            result_file_h = open(result_file_path_h, 'a')
            result_file_h.write(str_current_time + "\n")
            result_file_h.write(self.hostname + "\n")
            result_file_h.write(str_host_ip + "\n")
            result_file_h.write(str_free_hugepages + "\n")
            result_file_h.write(str_free_mem + "\n")
            result_file_h.write(str_mem_usage_percent + "\n")
            result_file_h.write(str_cpu_usage + "\n")

            host_process_list = HOST_PROCESS.split(";")
            for process in host_process_list:
                if len(process) == 0:
                    continue
                strcmd = "ps -ef | grep -w " + process + " | grep -v grep | awk '{print $2,$NF}'"
                result = os.popen(strcmd).read()
                if result and len(result) > 1:
                    result = result.strip("\n")
                    item_list = result.split("\n")
                    for item in item_list:
                        pid = item.split(" ")[0]
                        if process in ["slogd", "config_file"]:
                            pname = process + str(item_list.index(item))
                        else:
                            pname = process
                        strcmd = "pmap -d " + pid + " | grep \"000:00000\" | awk '{sum+=$2} END{print sum;}'"
                        sec_result = os.popen(strcmd).read()
                        sec_result = sec_result.strip('\n')
                        if sec_result:
                            str_process_mem = pname + "(KB):" + sec_result
                        else:
                            str_process_mem = pname + "(KB):0"
                        result_file_h.write(str_process_mem + "\n")

                        # lsof查看进程占用文件句柄数
                        strcmd = "lsof -p " + pid + " | wc -l"
                        sec_result = os.popen(strcmd).read()
                        sec_result = sec_result.strip('\n')
                        if sec_result:
                            str_process_handle = pname + "(handle):" + sec_result
                        else:
                            str_process_handle = pname + "(handle):0"

                        result_file_h.write(str_process_handle + "\n")

            result_file_h.close()

            int_sleep_seconds = check_interval - (int_current_seconds % check_interval)
            time.sleep(int_sleep_seconds)
