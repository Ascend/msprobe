#!/usr/bin/python3.7
# coding=gbk

import time
import socket
import os
from ctypes import *
import ascend_function
#from ssh import SSHConnection

class MonitorDevice:
    def __init__(self, device_index):
        self.time_header = "OutputTime:"

        self.result_file_name_first = 'monitor'
        self.result_file_name_postfix = '.runlog'
        self.archive_result_file_name_postfix = '.gz'

        self.dev_index = device_index

        self.dsmiMonitorlogname = "dsmi_warning_%d.log" % device_index

        #default value
        #self.libDsmi = cdll.LoadLibrary('/home/h00215183/DSMI/turing_dsmi_api.so')

    def get_filesize(self, filePath):
        fsize = os.path.getsize(filePath)
        return fsize

    def query_dsmi(self, dsmi_cost_time, dsmi_file, device_list, device_info, function, result_file=None, errcode=""):
        result_list = []
        # set param type and return type
        ff = function
        if function in [self.libDsmi.func_dsmi_get_device_list]:
            ff.argtypes = [c_char_p]
        elif function in [self.libDsmi.func_dsmi_get_device_ip_address]:
            ff.argtypes = [c_int, c_char_p]
        else:
            ff.argtypes = [c_int]
        ff.restype = c_char_p

        if function == self.libDsmi.func_dsmi_get_device_list:
            result = function(device_list.encode('utf-8'), dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function == self.libDsmi.func_dsmi_get_device_utilization_rate:
            devtype_list = [1, 2, 5, 6, 10]
            for dev_type in devtype_list:
                if dev_type == 1:
                    check_value = device_info["ddr_utiliza"]
                elif dev_type == 2:
                    check_value = device_info["aicore_utiliza"]
                elif dev_type == 5:
                    check_value = device_info["ddr_bw_utiliza"]
                elif dev_type == 6:
                    check_value = device_info["hbm_utiliza"]
                elif dev_type == 10:
                    check_value = device_info["hbm_bw_utiliza"]
                else:
                    check_value = None
                result = function(self.dev_index, dev_type, check_value, dsmi_cost_time)
                if result:
                    result_list.append(result)
        elif function == self.libDsmi.func_dsmi_get_aicpu_info:
            result = function(self.dev_index, device_info["aicpu_utiliza"], dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function == self.libDsmi.func_dsmi_get_device_ip_address:
            result = function(self.dev_index, device_info["roce_ip"].encode('utf-8'), dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function in [self.libDsmi.func_dsmi_get_phyid_from_logicid, self.libDsmi.func_dsmi_get_logicid_from_phyid]:
            result = function(self.dev_index, device_info["id"], dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function in [self.libDsmi.func_dsmi_get_hbm_info]:
            result = function(self.dev_index, device_info["hbm_temp"], device_info["hbm_freq"], dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function in [self.libDsmi.func_dsmi_get_memory_info]:
            result = function(self.dev_index, device_info["ddr_freq"], dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function in [self.libDsmi.func_dsmi_get_device_power_info]:
            result = function(self.dev_index, device_info["power"], dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function in [self.libDsmi.func_dsmi_get_device_temperature]:
            result = function(self.dev_index, device_info["temperature"], dsmi_cost_time)
            if result:
                result_list.append(result)
        elif function in [self.libDsmi.func_dsmi_get_ecc_info]:
            devtype_list = [0, 2]
            for dev_type in devtype_list:
                result = function(self.dev_index, dev_type, dsmi_cost_time)
                if result:
                    result_list.append(result)
        elif function in [self.libDsmi.func_dsmi_query_errorstring]:
            errcode = errcode.strip(".").strip()
            err_list = errcode.split(",")
            for err in err_list:
                if err == "":
                    continue
                err = eval(err)
                result = function(self.dev_index, err)
                if result:
                    result_list.append(result)
        else:
            result = function(self.dev_index, dsmi_cost_time)
            if result:
                result_list.append(result)

        if result_file:
            self.record_result(function, result_list, result_file)
        restr = self.record_dsmi(function, result_list, dsmi_file)
        return restr

    def record_result(self, function, result_list, result_file):

        # result check
        for res in result_list:
            # bytes convert to string
            res = res.decode()
            if function == self.libDsmi.func_dsmi_get_device_utilization_rate:
                if "ddr_bw_utiliza" not in res and "aicore_utiliza" not in res and "hbm_utiliza" not in res:
                    continue
            info_list = res.split("ret value = 0,")
            if len(info_list) > 1:
                value_list = info_list[-1].split(",")
                if function == self.libDsmi.func_dsmi_get_hbm_info:
                    for value in value_list:
                        value = value.strip().strip(".")
                        if value.startswith("hbm_size:"):
                            hbm_total = value.split(":")[-1].split("KB")[0]
                            str_hbm_tot = "HBMSize(KB):" + str(hbm_total)
                            result_file.write(str_hbm_tot + "\n")
                        elif value.startswith("hbm_usage:"):
                            hbm_usage = value.split(":")[-1].split("KB")[0]
                            str_hbm_usage = "HBMUsage(KB):" + hbm_usage
                            result_file.write(str_hbm_usage + "\n")
                        elif value.startswith("hbm_temp:"):
                            hbm_temp = value.split(":")[-1]
                            str_hbm_temp = "HbmTemperature(" + u'°C' + "):" + hbm_temp
                            result_file.write(str_hbm_temp + "\n")
                        elif value.startswith("hbm_bw:"):
                            hbm_bw_utiliza = value.split(":")[-1].split("%")[0]
                            str_hbm_bw_utiliza = "HbmBWUtiliza(%):" + hbm_bw_utiliza
                            result_file.write(str_hbm_bw_utiliza + "\n")
                        else:
                            pass
                elif function == self.libDsmi.func_dsmi_get_memory_info:
                    for value in value_list:
                        value = value.strip().strip(".")
                        if value.startswith("ddr_size:"):
                            ddr_total = value.split(":")[-1].split("KB")[0]
                            str_ddr_tot = "DDRSize(KB):" + ddr_total
                            result_file.write(str_ddr_tot + "\n")
                        elif value.startswith("ddr_utiliza:"):
                            ddr_utiliza = value.split(":")[-1].split("%")[0]
                            str_ddr_utiliza = "DDRUtiliza(%):" + str(ddr_utiliza)
                            result_file.write(str_ddr_utiliza + "\n")
                        else:
                            pass
                elif function == self.libDsmi.func_dsmi_get_device_temperature:
                    for value in value_list:
                        value = value.strip().strip(".")
                        if value.startswith("device_temperature"):
                            temperature = value.split("=")[-1]
                            str_temperature = "temperature(" + u'°C' + "):" + temperature
                            result_file.write(str_temperature + "\n")
                elif function == self.libDsmi.func_dsmi_get_device_power_info:
                    for value in value_list:
                        value = value.strip().strip(".")
                        if value.startswith("device_power"):
                            power = value.split("=")[-1].split("W")[0]
                            str_power = "power(W):" + power
                            result_file.write(str_power + "\n")
                elif function == self.libDsmi.func_dsmi_get_device_utilization_rate:
                    for value in value_list:
                        value = value.strip().strip(".")
                        if value.startswith("ddr_bw_utiliza"):
                            ddr_bw_utiliza = value.split("=")[-1].split("%")[0]
                            str_ddr_bw_utiliza = "DdrBWUtiliza(%):" + ddr_bw_utiliza
                            result_file.write(str_ddr_bw_utiliza + "\n")
                        elif value.startswith("aicore_utiliza"):
                            aicore_utiliza = value.split("=")[-1].split("%")[0]
                            str_aicore_utiliza = "AICoreUtiliza(%):" + aicore_utiliza
                            result_file.write(str_aicore_utiliza + "\n")
                        elif value.startswith("hbm_utiliza"):
                            hbm_utiliza = value.split("=")[-1].split("%")[0]
                            str_hbm_utiliza = "HBMUtiliza(%):" + hbm_utiliza
                            result_file.write(str_hbm_utiliza + "\n")
                        else:
                            pass
                elif function == self.libDsmi.func_dsmi_get_aicpu_info:
                    for value in value_list:
                        value = value.strip().strip(".")
                        if value.startswith("utilRate"):
                            aicpu_list = value.split("=")[-1].split(";")
                            index = 0
                            for aicpu_item in aicpu_list:
                                uti = aicpu_item.split("%")[0]
                                str_uti = "AICpuUtiliza" + str(index) + "(%):" + uti
                                result_file.write(str_uti + "\n")
                                index = index + 1
                else:
                    pass

            else:
                if function == self.libDsmi.func_dsmi_get_hbm_info:
                    str_hbm_tot = "HBMSize(KB):0"
                    str_hbm_usage = "HBMUsage(KB):0"
                    str_hbm_temp = "HbmTemperature(" + u'°C' + "):0"
                    str_hbm_bw_utiliza = "HbmBWUtiliza(%):0"
                    result_file.write(str_hbm_tot + "\n")
                    result_file.write(str_hbm_usage + "\n")
                    result_file.write(str_hbm_temp + "\n")
                    result_file.write(str_hbm_bw_utiliza + "\n")
                elif function == self.libDsmi.func_dsmi_get_memory_info:
                    str_ddr_tot = "DDRSize(KB):0"
                    str_ddr_utiliza = "DDRUtiliza(%):0"
                    result_file.write(str_ddr_tot + "\n")
                    result_file.write(str_ddr_utiliza + "\n")
                elif function == self.libDsmi.func_dsmi_get_device_temperature:
                    str_temperature = "temperature(" + u'°C' + "):0"
                    result_file.write(str_temperature + "\n")
                elif function == self.libDsmi.func_dsmi_get_device_power_info:
                    str_temperature = "power(W):0"
                    result_file.write(str_temperature + "\n")
                elif function == self.libDsmi.func_dsmi_get_device_utilization_rate:
                    str_ddr_bw_utiliza = "DdrBWUtiliza(%):0"
                    str_aicore_utiliza = "AICoreUtiliza(%):0"
                    str_hbm_utiliza = "HBMUtiliza(%):0"
                    result_file.write(str_ddr_bw_utiliza + "\n")
                    result_file.write(str_aicore_utiliza + "\n")
                    result_file.write(str_hbm_utiliza + "\n")
                else:
                    pass


    def record_dsmi(self, function, result_list, dsmi_file):

        # result check
        for res in result_list:
            # bytes convert to string
            res = res.decode()
            if "failed" in res:
                str_current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                line = "[{}][{}]{}".format(str_current_time, "ERROR", res)
                dsmi_file.write(line + "\n")

            elif "warning" in res:
                str_current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                line = "[{}][{}]{}".format(str_current_time, "WARN", res)
                dsmi_file.write(line + "\n")

            else:
                if function == self.libDsmi.func_dsmi_get_device_errorcode:
                    return res.split("errorcode=")[-1]

        return ""


    def query_device_info(self, dsmi_cost_time, dsmi_file, result_file, device_list, device_info):

        #device hbm info
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_hbm_info, result_file)
        #device memory info --DDR
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_memory_info, result_file)
        #device uti info
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_device_utilization_rate, result_file)
        #device temp
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_device_temperature, result_file)
        #device power
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_device_power_info, result_file)
        #device aicpu utirate
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_aicpu_info, result_file)

        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_device_health)
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_aicore_info)
        #self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_network_health)
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_phyid_from_logicid)
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_logicid_from_phyid)
        #self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_device_ip_address)
        #self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_chip_info)
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_ecc_info)
        errcode = self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_device_errorcode)
        self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_query_errorstring, errcode)
        if self.dev_index == 1:
            self.query_dsmi(dsmi_cost_time, dsmi_file, device_list, device_info, self.libDsmi.func_dsmi_get_device_list)


    def monitor_device(self, device_info, global_dict):

        check_interval = global_dict.get("check_interval", 0)
        archive_interval = global_dict.get("archive_interval", 0)
        int_max_size_per_runlog = global_dict.get("int_max_size_per_runlog", 0)
        int_max_archive_day_num = global_dict.get("int_max_archive_day_num", 0)
        device_list = global_dict.get("device_list", "")
        dsmi_sharelib_path = global_dict.get("dsmi_sharelib_path", "")
        # 250ms
        dsmi_cost_time = global_dict.get("dsmi_cost_time", 0)

        # get cfg path
        self.libDsmi = cdll.LoadLibrary(dsmi_sharelib_path)

        # init env
        str_log_folder_name_d = device_info.get("str_log_folder_name_d")
        result_file_name = self.result_file_name_first + self.result_file_name_postfix
        print(os.popen('mkdir -pv ' + str_log_folder_name_d).read())
        print(os.popen('rm -rf ' + str_log_folder_name_d + "/*").read())
        result_file_path_d = str_log_folder_name_d + "/" + result_file_name
        print(os.popen('touch ' + result_file_path_d).read())

        print(os.popen('rm -rf ' + self.dsmiMonitorlogname).read())
        print(os.popen('touch ' + self.dsmiMonitorlogname).read())


        # 设置初始归档Flag
        int_last_seconds = time.time()
        g_archive_flag = int(int_last_seconds / archive_interval)
        st_last_time = time.localtime(int_last_seconds)
        while True:
            int_current_seconds = time.time()
            st_current_time = time.localtime(int_current_seconds)

            # 判断是否需要打包压缩归档
            bool_need_archive_d = (st_last_time.tm_mday != st_current_time.tm_mday)

            archive_flag = int(int_current_seconds / archive_interval)
            if g_archive_flag:
                if g_archive_flag != archive_flag:
                    bool_need_archive_d = True


            # 判断devicer侧的日志是否超过日志文件最大size
            if (self.get_filesize(result_file_path_d)) > int_max_size_per_runlog:
                bool_need_archive_d = True

            if bool_need_archive_d:
                # 按照年月日创建目录
                str_archive_folder_name = str_log_folder_name_d + "/" + time.strftime("%Y%m%d", st_last_time)
                str_archive_file_name = self.result_file_name_first + '_' + time.strftime("%Y%m%d%H%M%S", st_last_time) + self.result_file_name_postfix
                print(os.popen('mkdir -pv ' + str_archive_folder_name).read())
                print(os.popen('mv ' + result_file_path_d + ' ' + str_archive_folder_name + '/' + str_archive_file_name + ';gzip -f ' + str_archive_folder_name + '/' + str_archive_file_name).read())

                # 超过90天的目录删除
                date_folder_names = os.listdir(str_log_folder_name_d)
                date_folder_names.sort()
                while len(date_folder_names) > int_max_archive_day_num:
                    old_folder = date_folder_names.pop(0)
                    print(os.popen('rm -rfv ' + str_log_folder_name_d + '/' + old_folder).read())


            # 重新赋值归档标识
            g_archive_flag = archive_flag
            st_last_time = st_current_time

            str_current_time = self.time_header + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int_current_seconds))

            result_file = open(result_file_path_d, 'a')
            result_file.write(str_current_time + "\n")

            # device侧搜集hbm内存信息+ddr内存信息
            dsmi_file = open(self.dsmiMonitorlogname, 'a')
            self.query_device_info(dsmi_cost_time, dsmi_file, result_file, device_list, device_info)
            dsmi_file.close()

            result_file.close()

            int_sleep_seconds = check_interval - (int_current_seconds % check_interval)
            time.sleep(int_sleep_seconds)


class MonitorDeviceSSH:
    def __init__(self):
        self.time_header = "OutputTime:"
        self.hostname_header = "HostName:"
        self.ip_header = "HostIPPort:"
        self.free_hugepages_header = "FreeHugePages:"
        self.free_mem_header = "FreeMem(MB):"
        self.mem_usage_percent_header = "MemUsage(%):"
        self.cpu_usage_header = "CpuUsage(%):"

        self.result_file_name_first = 'monitor'
        self.result_file_name_postfix = '.runlog'
        self.archive_result_file_name_postfix = '.gz'

        self.hostname = self.hostname_header + socket.gethostname()

    def get_filesize(self, filePath):
        fsize = os.path.getsize(filePath)
        return fsize

    def monitor_device_ssh(self, device_info, global_dict):

        check_interval = global_dict.get("check_interval", 0)
        archive_interval = global_dict.get("archive_interval", 0)
        int_max_size_per_runlog = global_dict.get("int_max_size_per_runlog", 0)
        int_max_archive_day_num = global_dict.get("int_max_archive_day_num", 0)
        DEVICE_PROCESS = global_dict.get("DEVICE_PROCESS", "")

        ip = device_info.get("ip")
        port = device_info.get("port")
        username = device_info.get("username")
        password = device_info.get("password")

        # init env
        str_log_folder_name_d = device_info.get("str_log_folder_name_d")
        result_file_name = self.result_file_name_first + self.result_file_name_postfix
        print(os.popen('mkdir -pv ' + str_log_folder_name_d).read())
        print(os.popen('rm -rf ' + str_log_folder_name_d + "/*").read())
        result_file_path_d = str_log_folder_name_d + "/" + result_file_name
        print(os.popen('touch ' + result_file_path_d).read())

        conn = SSHConnection(ip, port, username, password)

        # 设置初始归档Flag
        int_last_seconds = time.time()
        g_archive_flag = int(int_last_seconds / archive_interval)
        st_last_time = time.localtime(int_last_seconds)
        while True:
            int_current_seconds = time.time()
            st_current_time = time.localtime(int_current_seconds)

            # 判断是否需要打包压缩归档
            bool_need_archive_d = (st_last_time.tm_mday != st_current_time.tm_mday)

            archive_flag = int(int_current_seconds / archive_interval)
            if g_archive_flag:
                if g_archive_flag != archive_flag:
                    bool_need_archive_d = True


            # 判断devicer侧的日志是否超过日志文件最大size
            if (self.get_filesize(result_file_path_d)) > int_max_size_per_runlog:
                bool_need_archive_d = True

            if bool_need_archive_d:
                # 按照年月日创建目录
                str_archive_folder_name = str_log_folder_name_d + "/" + time.strftime("%Y%m%d", st_last_time)
                str_archive_file_name = self.result_file_name_first + '_' + time.strftime("%Y%m%d%H%M%S", st_last_time) + self.result_file_name_postfix
                print(os.popen('mkdir -pv ' + str_archive_folder_name).read())
                print(os.popen('mv ' + result_file_path_d + ' ' + str_archive_folder_name + '/' + str_archive_file_name + ';gzip -f ' + str_archive_folder_name + '/' + str_archive_file_name).read())

                # 超过90天的目录删除
                date_folder_names = os.listdir(str_log_folder_name_d)
                date_folder_names.sort()
                while len(date_folder_names) > int_max_archive_day_num:
                    old_folder = date_folder_names.pop(0)
                    print(os.popen('rm -rfv ' + str_log_folder_name_d + '/' + old_folder).read())


            # 重新赋值归档标识
            g_archive_flag = archive_flag
            st_last_time = st_current_time

            str_current_time = self.time_header + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int_current_seconds))

            result_file = open(result_file_path_d, 'a')
            result_file.write(str_current_time + "\n")
            result_file.write(self.hostname + "\n")
            str_ip = self.ip_header + device_info.get("ip") + '_' + str(device_info.get("port"))
            result_file.write(str_ip + "\n")

            if conn:
                # device侧执行收集内存命令
                result = conn.exec_command("cat /proc/meminfo | grep HugePages_Free | awk '{print $2}'")
                str_free_hugepages = self.free_hugepages_header + result.decode().strip('\n')

                # device侧执行搜集CPU命令
                result = conn.exec_command(
                    "/usr/sbin/mpstat -P ALL 1 1 | grep \" all \" | grep -v \"Average:\" | awk '{print 100-$11}'")
                str_cpu_usage = self.cpu_usage_header + result.decode().strip('\n')

                result_file.write(str_free_hugepages + "\n")
                result_file.write(str_cpu_usage + "\n")

                cmds = device_info.get("command", {})
                for k, v in cmds.items():
                    try:
                        result_file.write(k + ":" + conn.exec_command(v).decode() + "\n")
                    except:
                        print("can not get value for %s" % k)

                # query process mem info
                device_process_list = DEVICE_PROCESS.split(";")
                for process in device_process_list:
                    if len(process) == 0:
                        continue
                    # 58617 HwHiAiUs  1d17 compute_process --deviceId=0 --pid=55551 --profilingMode=0
                    # 58618 HwHiAiUs  0:21 hccp_service.bin --deviceId=0 --pid=55551
                    if process in ["compute_process", "hccp_service.bin"]:
                        strcmd = "/usr/sbin/ps -ef | grep -w " + process + " | grep -v grep | awk '{print $1,$5}'"
                    else:
                        strcmd = "/usr/sbin/ps -ef | grep -w " + process + " | grep -v grep | awk '{print $1,$NF}'"
                    result = conn.exec_command(strcmd)
                    if result and len(result) > 1:
                        result = result.decode().strip("\n")
                        item_list = result.split("\n")
                        for item in item_list:
                            pid = item.split(" ")[0]
                            if process in ["compute_process", "hccp_service.bin"]:
                                tmp_id = item.split(" ")[-1].split("=")[-1]
                                pname = process + tmp_id
                            else:
                                pname = process
                            strcmd = "/usr/sbin/pmap " + pid + " | grep \"\[\" | awk '{sum+=substr($2,1,length($2)-1)} END{print sum;}'"
                            sec_result = conn.exec_command(strcmd)
                            sec_result = sec_result.decode().strip('\n')
                            if sec_result:
                                str_process_mem = pname + "(KB):" + sec_result

                            else:
                                str_process_mem = pname + "(KB):" + "0"

                            result_file.write(str_process_mem + "\n")

                            # lsof查看进程占用文件句柄数
                            strcmd = "/usr/sbin/lsof | grep " + pid + " | /usr/sbin/wc -l"
                            sec_result = conn.exec_command(strcmd)
                            sec_result = sec_result.decode().strip('\n')
                            if sec_result:
                                str_process_handle = pname + "(handle):" + sec_result
                            else:
                                str_process_handle = pname + "(handle):0"

                            result_file.write(str_process_handle + "\n")

            result_file.close()

            int_sleep_seconds = check_interval - (int_current_seconds % check_interval)
            time.sleep(int_sleep_seconds)


