#!/usr/bin/python3.7
# coding=gbk

import multiprocessing
import configparser
from monitor_device import MonitorDevice, MonitorDeviceSSH
from monitor_host import MonitorHost
#from bak_log import BakLog


def get_cfg_info():
    config = configparser.ConfigParser()
    config.read_file(open('monitor.ini'))

    global_dict = {}
    global_dict.update(check_interval=int(config.get("DEFAULT", "checkinterval")),
                       archive_interval=int(config.get("DEFAULT", "archiveinterval")),
                       log_check_interval=int(config.get("DEFAULT", "logcheckinterval")),
                       int_max_size_per_runlog=int(config.get("DEFAULT", "max_size_per_runlog")),
                       int_max_archive_day_num=int(config.get("DEFAULT", "max_archive_day_num")),
                       SYS_LOG_PATH=config.get("DEFAULT", "syslogpath"),
                       NETWORK_CARD=config.get("DEFAULT", "networkcard"),
                       HOST_PROCESS=config.get("DEFAULT", "hostprocessname"),
                       DEVICE_PROCESS=config.get("DEFAULT", "deviceprocessname"),
                       dsmi_cost_time=int(config.get("DEFAULT", "dsmi_cost_time")),
                       device_list=config.get("DEFAULT", "device_list"),
                       dsmi_sharelib_path=config.get("DEFAULT", "dsmi_sharelib_path"))

    sections_list = config.sections()
    print(sections_list)

    device_info_list = []
    for section in sections_list:
        device_dict = dict()

        log_dir = config.get(section, "log_dir")
        temperature = int(config.get(section, "temperature"))
        power = int(config.get(section, "power"))
        ddr_utiliza = int(config.get(section, "ddr_utiliza"))
        aicore_utiliza = int(config.get(section, "aicore_utiliza"))
        ddr_bw_utiliza = int(config.get(section, "ddr_bw_utiliza"))
        hbm_utiliza = int(config.get(section, "hbm_utiliza"))
        hbm_bw_utiliza = int(config.get(section, "hbm_bw_utiliza"))
        ddr_freq = int(config.get(section, "ddr_freq"))
        hbm_freq = int(config.get(section, "hbm_freq"))
        hbm_temp = int(config.get(section, "hbm_temp"))
        aicpu_utiliza = int(config.get(section, "aicpu_utiliza"))
        id = int(config.get(section, "id"))
        #roce_ip = config.get(section, "roce_ip")

        if config.has_option(section, "ip"):
            device_ssh_dict = dict()
            ip = config.get(section, "ip")
            username = config.get(section, "username")
            password = config.get(section, "password")
            port = int(config.get(section, "port"))
            cmds = {}
            for key in config[section]:
                if key.startswith("cmd_"):
                    str_cmd_name = key[len("cmd_"):]
                    cmds[str_cmd_name] = config.get(section, key)

            print(cmds)

            ssh_log_dir = log_dir + "_ssh"
            device_ssh_dict.update(ip=ip, username=username, password=password, port=port,
                               str_log_folder_name_d=ssh_log_dir, command=cmds)

            device_info_list.append(device_ssh_dict)

        device_dict.update(str_log_folder_name_d=log_dir, temperature=temperature, power=power, ddr_utiliza=ddr_utiliza,
                           ddr_bw_utiliza=ddr_bw_utiliza, hbm_utiliza=hbm_utiliza, hbm_bw_utiliza=hbm_bw_utiliza,
                           ddr_freq=ddr_freq, hbm_freq=hbm_freq, hbm_temp=hbm_temp, id=id,
                           aicore_utiliza=aicore_utiliza, aicpu_utiliza=aicpu_utiliza)

        device_info_list.append(device_dict)

    return global_dict, device_info_list

def _monitor_device(index, device_info, global_dict):
    md = MonitorDevice(index)
    md.monitor_device(device_info, global_dict)

def _monitor_device_ssh(device_info, global_dict):
    md = MonitorDeviceSSH()
    md.monitor_device_ssh(device_info, global_dict)

def _monitor_host(global_dict):
    mh = MonitorHost()
    mh.monitor_host(global_dict)

#def _bak_log(global_dict):
#    bl = BakLog()
#    bl.bak_log(global_dict)


if __name__ == "__main__":

    # init env
    global_dict, device_info_list = get_cfg_info()

    # device mem and cpu
    process_list = []
    for device_info in device_info_list:
        if "id" in device_info:
            index = int(device_info.get("id"))
            p = multiprocessing.Process(target=_monitor_device, args=(index, device_info, global_dict,))
        else:
            p = multiprocessing.Process(target=_monitor_device_ssh, args=(device_info, global_dict,))
        process_list.append(p)
        p.start()

    # host mem and cpu
    p = multiprocessing.Process(target=_monitor_host, args=(global_dict,))
    process_list.append(p)
    p.start()

    # bak log
    #p = multiprocessing.Process(target=_bak_log, args=(global_dict,))
    #process_list.append(p)
    #p.start()

    for p in process_list:
        p.join()




