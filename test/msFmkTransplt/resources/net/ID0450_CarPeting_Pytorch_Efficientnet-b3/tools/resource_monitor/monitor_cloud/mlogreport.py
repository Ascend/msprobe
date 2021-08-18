#!/usr/bin/python3
# coding=gbk
# Analyse runlog and export it to a excel
# Support single core and total core.
import os
import sys
import configparser
import time
import datetime
import xlsxwriter
import re


global UNKNOW
UNKNOW       = 0
global PartOnly  #0x01
PartOnly = 1  #0x01
global MainOnly 
MainOnly = 2 #0x10
global BOTH
BOTH     = PartOnly | MainOnly
global core_num
core_num = 32
global excelarea
excelarea = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
            'Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF',
            'AG','AH','AI','AJ','AK','AL','AM','AN']
global core32title
core32title = ["time", "Total", "Core_0", "Core_1", "Core_2", "Core_3", "Core_4",
          "Core_5","Core_6","Core_7","Core_8","Core_9","Core_10","Core_11",
          "Core_12","Core_13","Core_14","Core_15","Core_16","Core_17","Core_18",
          "Core_19","Core_20","Core_21","Core_22","Core_23","Core_24","Core_25",
          "Core_26","Core_27","Core_28","Core_29","Core_30","Core_31"]

global report_folder_host
report_folder_host = "report_host"

global mainctrltitle  
mainctrltitle = ["time", "Total"]

global outfileName
ISOTIMEFORMAT = '%Y%m%d%H%M%S'

outfileName = "Result_%s.xlsx" % time.strftime(ISOTIMEFORMAT, time.localtime())

class CoreList:
    def __init__(self): 
        #0 - 31 is each core.
        # each core coreid--->value
        self.core = dict()
        self.total = 0
    def printlog(self, datatype):
        if datatype == MainOnly:
            print("total:", self.total)
        elif datatype == PartOnly:
            for i in range(0,core_num):
                if self.core.has_key(i):            
                    self.total = self.total + self.core[i]
                else:
                    print("core[%d]: 0" % i)
            print(self.total)
            for i in range(0,core_num):
                if self.core.has_key(i):
                    print("core[%d]:" % i, self.core[i])
                else:
                    print("core[%d]: 0" % i)
            
        elif datatype == BOTH:
            print("total:", self.total)
            for i in range(0,core_num):
                if self.core.has_key(i):
                    print("core[%d]:" % i, self.core[i])
                else:
                    print("core[%d]: 0" % i)
        else :
            print("unkonw data")
    
    def get_total(self, datatype):  
        if datatype == PartOnly:
            for i in range(0, core_num):
                if self.core.has_key(i):
                    self.total = self.total + self.core[i]
        elif datatype == UNKNOW:
            print("unknow data")
            return 0
                    
            
        return self.total
    
class CKeyResList:
    '''
        keytype:
            0x00: unknow
            0x01: each core have one data
            0x10: just main_header
            0x11: mixed
    '''
    def __init__(self):
        # timestamp ---> CoreList
        self.datalist = dict()
        self.keytype = UNKNOW       
    
    def printlog(self, timestamp):
        print("Type:", self.keytype)
        self.datalist[timestamp].printlog(self.keytype)
    def adddefault(self, timestamp):
        self.datalist.setdefault(timestamp, CoreList())
        self.datalist[timestamp].total = 0


class CResultSet:
    '''
    '''
    def __init__(self, begin_time, end_time):
        self.timestamplist = list()
        # keyword --> CKeyResList
        self.keyReslist = dict()
        self.begin_time = begin_time
        self.end_time = end_time

    def printlog(self):
        for keywords in self.keyReslist.keys():
            print("keywords", keywords)
            for timeitem in self.timestamplist:
                print("timestamp:", timeitem)
                self.keyReslist[keywords].printlog(timeitem)

    def inittable(self, tablefd, dataType):
        
        if dataType != MainOnly: 
            tablefd.write_row(0,0, core32title[0:core_num+2])
        else:
            tablefd.write_row(0,0, mainctrltitle)

    def output(self, FileName, report_folder):

        strtime = FileName.split("_")[-1].split(".")[0]
        fileName = "Result_cal_%s.txt" % strtime
        calfile = os.path.join(os.getcwd(), report_folder + "/" + fileName)
        fw = open(calfile, "w+")

        #open excel
        docname   = os.path.normpath(os.path.join(os.getcwd(), report_folder + "/" + FileName))
        exceldoc = xlsxwriter.Workbook(docname)
        date_format = exceldoc.add_format({'num_format': 'hh:mm:ss'})
        #遍历项目
        for keywords in self.keyReslist.keys():
            #create sheet for every key
            try:
                keyTable = exceldoc.add_worksheet(keywords)
            except :
                keyTable = exceldoc.add_worksheet()

            sheetName = keyTable.get_name()
            rowIndex = 0
            keytype = self.keyReslist[keywords].keytype

            sum = 0
            max = 0
            min = 999999999999999999

            #遍历时间戳
            for timeitem in self.timestamplist:
                if timeitem < self.begin_time:
                    continue

                if timeitem > self.end_time:
                    break

                if timeitem not in self.keyReslist[keywords].datalist:
                    self.keyReslist[keywords].adddefault(timeitem)
                every = self.keyReslist[keywords].datalist[timeitem].get_total(keytype)
                sum += every
                if every > max:
                    max = every
                if every < min:
                    min = every

                colIndex = 0
                if rowIndex == 0:
                    self.inittable(keyTable, keytype)
                    rowIndex = rowIndex + 1

                #time
                #keyTable.write(rowIndex, colIndex, timeitem)
                time_4_write = datetime.datetime.strptime(timeitem, "%Y-%m-%d %H:%M:%S")
                keyTable.write_datetime(rowIndex, colIndex, time_4_write, date_format)
                colIndex = colIndex + 1             
                #total write in the last
                keyTable.write(rowIndex, colIndex, every)
                colIndex = colIndex + 1
                #each core
                coredata = self.keyReslist[keywords].datalist[timeitem].core
                if keytype == PartOnly or keytype == BOTH:
                    for coreid in range(0, core_num):
                        if coredata.has_key(coreid):
                            keyTable.write(rowIndex, colIndex, coredata[coreid])
                        else:
                            keyTable.write(rowIndex, colIndex, 0)
                        colIndex = colIndex + 1 
                rowIndex = rowIndex + 1

            #calc max min avg
            if len(self.timestamplist) > 0:
                avg = round(sum / len(self.timestamplist), 2)
            else:
                avg = 0.00

            line = keywords + ":" + str(avg) + "," + str(max) + "," + str(min)
            fw.write(line)
            fw.write("\n")
            
            #time-->total:
            charttotal = exceldoc.add_chart({'type': 'line'})
            categories = '=%s!$A$2:$A$%d' % ("'" + sheetName + "'", rowIndex)
            dataset    = '=%s!$B$2:$B$%d' % ("'" + sheetName + "'", rowIndex)
            seriesname = '=%s!$B$1' % ("'" + sheetName + "'")
            charttotal.add_series({
                'categories': categories,
                'values': dataset,
                'name'  : seriesname,
                })
            charttotal.set_title ({'name': keywords + "(" + self.begin_time + "~" + self.end_time + ")"})
            charttotal.set_x_axis({'name': 'time'})
            charttotal.set_size({'width': 1280, 'height': 500})

            keyTable.insert_chart('C5', charttotal, {'x_scale': 1, 'y_scale': 1})
            
            #time---->each core
            if keytype == PartOnly or keytype == BOTH:
                chartcore = exceldoc.add_chart({'type': 'line'})
                chartcore.set_title ({'name': keywords})  
                chartcore.set_x_axis({'name': 'time'})          
                for i in range(0,  core_num):           
                    dataset    = '=%s!$%s$2:$%s$%d' % ("'" + sheetName + "'", excelarea[i + 2], excelarea[i + 2], rowIndex)         
                    seriesname = '=%s!$%s$1' % ("'" + sheetName + "'", excelarea[i + 2])
                    chartcore.add_series({
                        'categories': categories,
                        'values': dataset,
                        'name'  : seriesname,
                        })              
                keyTable.insert_chart('K20', chartcore)

        exceldoc.close()
        fw.close()

        return docname

class CSPURunlog:
    def __init__(self, begin_time, end_time):
        self.FileList = list() #files to search
        self.keyList  = list() #key words list
        self.ResultList = CResultSet(begin_time, end_time) #result, store CResultSet
        self.FileName = outfileName

    def setFileList(self, files):
        '''
            get full path of each file
        '''
        for file in files:
            filepath = os.path.normpath(os.path.join(os.getcwd(), file))
            if not os.path.exists(filepath):
                print("Note: %s is not exist, ignore it" % file)
                print("Note: fullpath %s is not exist, ignore it" % filepath)
            else:
                self.FileList.append(filepath)
        self.FileList.sort()

    def setOutputFile(self, keyValue):
        '''
            get output filename
        '''
        #delete invalid string
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/\:*?"<>|'       
        new_title = re.sub(rstr, "", keyValue)
        if len(keyValue) > 50:
            new_title = keyValue[:50]
        self.FileName = "Result_%s_%s.xlsx" % (new_title, time.strftime(ISOTIMEFORMAT, time.localtime()))       
        print(self.FileName)
        
    def setKeyList(self, keys):
        '''
            get key list of each file
        '''
        #first , add some string ino list. because it will be need in all time
        #and these should be the first three key 
        fileKeyName = ""
        if len(keys[0]) == 0:
            print("gather default data")
            fileKeyName = r"All"
        else:
            for key in keys:
                print("keywords:%s"%(key))
                if fileKeyName == "":
                    fileKeyName = key
                    self.keyList.append(key)
        self.setOutputFile(fileKeyName)
    
    def getcmd(self, fileName):
        cmdstr = ''
        for key in self.keyList:
            if key.strip() == '':
                continue
                
            if cmdstr == '':
                cmdstr = key
            else:
                cmdstr = cmdstr + '|' + key
        global output
        output = r'OutputTime:'
        hostname_header = r'HostName'
        hostipport_header = r'HostIPPort'
        if cmdstr == '':
            cmdstr = 'egrep -v "%s"' % ('^$' + '|' + hostname_header + '|' + hostipport_header)
        else:
            cmdstr = cmdstr + '|' + output
            cmdstr = 'egrep "%s"' %cmdstr
        if fileName.split('.')[-1] == 'gz':
            cmdstr = 'zcat %s|' % fileName + cmdstr
        else:
            cmdstr = 'cat %s|' % fileName + cmdstr
        print(cmdstr)
        return cmdstr

    def zero(self, seqbegin, seqend):
        print("zero thread[%d] to thread[%d], mybe runlog is wrong" % (seqbegin, seqend))
        
    def ProcRunlog(self):
        for fileName in self.FileList:
            cmdstr = self.getcmd(fileName)
            if cmdstr == '':
                print("wrong input")
                Usage()
                exit(0)
            
            retCode = 0
            retstr = os.popen(cmdstr).read()
            if retCode == 0:
                retstr.split("\n")
                datatype = MainOnly
                for line in retstr.splitlines():
                    # begin a new segment
                    if line.startswith(output):
                        timestamp = line[line.find(':') + 1:]
                        self.ResultList.timestamplist.append(timestamp)
                    else:
                        #get key words from result
                        keyrightpos = line.rfind(':')
                        if keyrightpos == -1:
                            print("get keywords failed ", line)
                            exit(0)
                        keywords = line[0:keyrightpos]
                        valueLeftpos = keyrightpos
                        if valueLeftpos == -1:
                            print("get value failed ", line)
                            exit(0)

                        try:
                            value = eval(line[valueLeftpos + 1:])
                            if type(value) not in [int, float]:
                                value = 0
                                print("%s" % timestamp)
                                print("%s" % line)
                        except:
                            value = 0
                            print("%s" % timestamp)
                            print("%s" % line)

                        self.ResultList.keyReslist.setdefault(keywords, CKeyResList())
                        self.ResultList.keyReslist[keywords].datalist.setdefault(timestamp, CoreList())
                        self.ResultList.keyReslist[keywords].keytype = self.ResultList.keyReslist[keywords].keytype|datatype
                        self.ResultList.keyReslist[keywords].datalist[timestamp].total = value
                
            else:
                print("excute failed", cmdstr, retstr)
                    
    def printlog(self):
        self.ResultList.printlog()

    def output(self, report_folder):
        return self.ResultList.output(self.FileName, report_folder)

def Usage():
    '''
    ./mlogreport.py [KeyWords] [FileLists]

    e.g:    
    1. Gather specified item from specified runlog file 
    ./mlogreport.py "MemUsage" "2018-09-20 00:00:00" "2018-09-20 14:00:00"
    
    2. Gather specified items from some runlog file
    ./mlogreport.py "FreeMem|MemUsage" "2018-09-20 00:00:00" "2018-09-20 14:00:00"
    
    3. Gather all important items from runlog file
    ./mlogreport.py "" "2018-09-20 00:00:00" "2018-09-20 14:00:00"
    '''
    print(Usage.__doc__)


str_log_folder_name_h = "hlog"
result_file_name_first = 'monitor'
result_file_name_postfix = '.runlog'
archive_result_file_name_postfix = '.gz'

def get_device_logdir():
    config = configparser.ConfigParser()
    config.read_file(open('monitor.ini'))

    sections_list = config.sections()
    print(sections_list)

    device_logdir_list = []
    for section in sections_list:
        log_dir = config.get(section, "log_dir")
        device_logdir_list.append(log_dir)
        if config.has_option(section, "ip"):
            device_logdir_list.append(log_dir + "_ssh")

    return device_logdir_list

def GetFileList(begin_time, end_time, str_log_folder_name):
    result = []
    # 查找所有需要扫描的文件
    date_begin = begin_time[0:8]
    print(date_begin)
    date_end = end_time[0:8]
    print(date_end)

    file_name_begin = result_file_name_first + "_" + begin_time + result_file_name_postfix + archive_result_file_name_postfix
    file_name_end = result_file_name_first + "_" + end_time + result_file_name_postfix + archive_result_file_name_postfix
    print(file_name_begin)
    print(file_name_end)

    # 文件夹
    date_folder_names = os.listdir(str_log_folder_name)
    date_folder_names.sort()
    for date_folder_name in date_folder_names:
        str_date_folder_path = str_log_folder_name + "/" + date_folder_name
        if (not os.path.isdir(str_date_folder_path)):
            continue
        # 20180920
        if date_folder_name < date_begin or date_folder_name > date_end:
            continue
        log_file_names = (os.listdir(str_date_folder_path))
        log_file_names.sort()
        for log_file_name in log_file_names:
            str_log_file_path = str_date_folder_path + "/" + log_file_name
            #print(str_log_file_path)
            if (not os.path.isfile(str_log_file_path)):
                continue
            # monitor_20180920114000.runlog.gz
            if log_file_name < file_name_begin:
                #print(log_file_name, "small", file_name_begin)
                continue
            result.append(str_log_file_path)
            #print(log_file_name, "append")
            if  log_file_name > file_name_end:
                #print(log_file_name, "big", file_name_end)
                # 文件名时间对应的是记录结束时间，所以应该往后多取一个
                break

    return result


def gen_report(str_log_folder_name, report_folder):

    #begin_time = "20180920000000"
    output_begin_time =sys.argv[2]
    #end_time = "20180920140000"
    output_end_time = sys.argv[3]
    # 检查时间格式

    begin_time = time.strftime("%Y%m%d%H%M%S", (time.strptime(output_begin_time, "%Y-%m-%d %H:%M:%S")))
    end_time = time.strftime("%Y%m%d%H%M%S", (time.strptime(output_end_time, "%Y-%m-%d %H:%M:%S")))

    files = GetFileList(begin_time, end_time, str_log_folder_name)
    # 添加正在生成的日志
    files.append(str_log_folder_name + "/" + result_file_name_first + result_file_name_postfix)
    # print(files)

    os.popen("mkdir -pv " + report_folder)
    # Begin
    print("Analyse %s begin" % report_folder)
    # main()
    print(output_begin_time)
    print(output_end_time)
    spurun = CSPURunlog(output_begin_time, output_end_time)
    spurun.setKeyList(sys.argv[1:2])
    spurun.setFileList(files)
    spurun.ProcRunlog()
    # spurun.printlog()
    result_file = spurun.output(report_folder)
    print("Analyse %s end" % report_folder)
    print("===========")
    print(result_file)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        Usage()
        exit(1)

    device_logdir_list = get_device_logdir()
    # 生成device报告
    for device_logdir in device_logdir_list:
        report_folder_device = "report_device_" + device_logdir
        gen_report(device_logdir, report_folder_device)

    # 生成host报告
    gen_report(str_log_folder_name_h, report_folder_host)

