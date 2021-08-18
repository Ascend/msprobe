import os
import sys
import time
import csv
import ascend_function

def check_performance(path):
    #print("##########################FPS_RESULT############################")
    item_list=[]
    file_FPS = open(path+".txt")
    List_FPS = []
    sum = 0
    for line in file_FPS:
        number = line.split("\n")
        FPS = float(number[0])
        List_FPS.append(FPS)
    file_FPS.close()
    del(List_FPS[0:2])
    num = len(List_FPS)
    for i in range(num):
        sum += float(List_FPS[i])
    average =round( sum / num)
    max_FPS =round( max(List_FPS),2)
    min_FPS = round(min(List_FPS),2)
    #print('average:%s,max_FPS:%s,min_FPS:%s'%(average,max_FPS,min_FPS))       
    result_list={"average":average,"max_FPS":max_FPS,"min_FPS":min_FPS} 
    time_local = (time.strftime("%Y%m%d-%H:%M:%S", time.localtime()))
    print_time = '[' + time_local + ']' + ' [INFO]' +str(result_list)
    print(print_time)
    #return print_time
    str_path=path.split('/')
    #FPS_path="/"+str_path[1]+"/"+str_path[2]+"/"+str_path[3]+"/"+str_path[4]
    casename=str_path[-1]
    del str_path[-2:]
    new_str=''
    for item in str_path:
        item=item+'/'
        new_str+=item
    os.system("touch " +new_str + "/FPS.csv")
    head_List = ["CaseName", "average_FPS", "max_FPS", "min_FPS"]
    # noinspection PyListCreation
    body_List = []
    body_List.append(casename)
    body_List.append(average)
    body_List.append(max_FPS)
    body_List.append(min_FPS)
    with open(new_str+"/FPS.csv", 'r+') as f:
        csvwrite = csv.writer(f)
        csvwrite.writerow(head_List)
    f.close()
    with open(new_str+"/FPS.csv",'a+',newline='') as f:
        csvwrite = csv.writer(f)
        csvwrite.writerow(body_List)
    f.close()
if __name__ == '__main__':
    path = sys.argv [1]
    check_performance(path)
