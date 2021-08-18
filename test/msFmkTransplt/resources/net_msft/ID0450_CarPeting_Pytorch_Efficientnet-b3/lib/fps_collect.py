import sys
import csv
import re
import ascend_function


list_im = []
line_w = []
date = []
list1 = []
list_out = []


def get_in():
    log_name = sys.argv[1]
    list_im.append(log_name)
    loop = sys.argv[2]
    list_im.append(loop)
    key_word = sys.argv[3]
    list_im.append(key_word)
    fen_ge = sys.argv[4]
    list_im.append(fen_ge)
    num = sys.argv[5]
    list_im.append(num)
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
    with open(line_w, 'r') as f_st:
        for line in f_st.readlines():
            if list_im[2] in line:
                list1.append(line.split(list_im[3])[int(list_im[4])])
        for i in list1:
            list2.append(re.findall(r'\d+\.*\d*',i))
        for a in range(len(list2)):
            for b in range(len(list2[a])):
                list_out.append(float(list2[a][b]))

				
def sort_and_comp(list):
    sum = 0.0
    #list_head = ['net_name','MAX','MIN','AVG']
    list_body = []
    name_net = list_im[0].split('/')[-1]
    name = name_net.split('.')[0]
    list_body.append(name)
    max_v = max(list[2:])
    list_body.append(max_v)
    min_v = min(list[2:])
    list_body.append(min_v)
    #file_ta = list_im[0].replace(name_net,"") + 'accuracy.csv'
    for i in range(2,len(list)):
        sum += list[i]
    avg = int(sum/(len(list) - 2))
    list_body.append(avg)
    print("MAX is    :",max_v)
    print("MIN is    :",min_v)
    print("AVERAGE is:",avg)

#write in .csv file
	#with open(file_ta,"w+") as f:
	#	writer = csv.writer(f)
	#	writer.writerow(list_head)
	#	writer.writerow(list_body)
	#	f.close()


if __name__ == '__main__':
    get_in()
    func_job()
    sort_and_comp(list_out)