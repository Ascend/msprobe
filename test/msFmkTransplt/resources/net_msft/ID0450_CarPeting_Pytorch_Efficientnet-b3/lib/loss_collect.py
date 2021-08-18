import sys
import csv
import re
import ascend_function


list_im = []
line_w = []
line_t = []
date = []
list1 = []
list_out = []



def get_in():
    log_name = sys.argv[1]
    list_im.append(log_name)
    #loop = sys.argv[2]
    #list_im.append(loop)
    key_word = sys.argv[2]
    list_im.append(key_word)
    fen_ge = sys.argv[3]
    list_im.append(fen_ge)
    num = sys.argv[4]
    list_im.append(num)
    filtration = "I0"
    list_im.append(filtration)
    return list_im

	
def func_job():
    global line_w,date,list1,list_out,line_t
    list2 = []
    list_device = []
    device_num = 0
    f = open(list_im[0])
    for line in f.readlines():
        if '[INFO] DEVICE_ID' in line:
            line = line.strip('\n')
            line = line.split('=')[1]
            list_device.append(line.split(' '))
            for i in range(len(list_device)):
                for o in range(len(list_device[0])):
                    device_num = list_device[0][1]
        if 'host mount dir:' in line:
            line_t = line.strip('\n')
    line_p = line_t + '/train_'+ str(device_num) +'.log'
    line_w = line_p.split(': ')[-1]
    f.close()
    if list_im[0].find('resnet50_ei') != -1:
        with open(line_w, 'r') as f_st:
            for line in f_st.readlines():
                if list_im[1] in line:
                    tmp = line.split(list_im[2])[int(list_im[3])]
                    list1.append(tmp.split("(")[0])
            for i in list1:
                list2.append(re.findall(r'\d+\.*\d*', i))
            for a in range(len(list2)):
                for b in range(len(list2[a])):
                    print(list2[a][b])
                    list_out.append(float(list2[a][b]))
    else:
        with open(line_w, 'r') as f_st:
            for line in f_st.readlines():
                if list_im[0].find('bert_nz') != -1 or list_im[0].find('deeplabv3_tf') != -1:
                    if list_im[1] in line and list_im[2] in line:
                        list1.append(line.split(list_im[2])[int(list_im[3])])
                else:
                    if list_im[1] in line and list_im[4] not in line and list_im[2] in line:
                        list1.append(line.split(list_im[2])[int(list_im[3])])
            for i in list1:
                result = re.findall(r'\d+\.*\d*e\+*\d*|\d+\.*\d*', i)
                print(result[0])
                list_out.append(result[0])
            #for a in range(len(list2)):
                #for b in range(len(list2[a])):
                #    print(list2[a][b])
                #list_out.append(float(list2[a][b]))
    print("the end of loss:", list_out[-1])


if __name__ == '__main__':
    get_in()
    func_job()
