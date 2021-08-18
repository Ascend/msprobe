# -*- coding:utf-8 -*-
import xlrd
import ascend_function

loop = 100
filename = 'training_trace_tag.0.xlsx'
data = xlrd.open_workbook(filename)
table = data.sheets()[0]
ncols = table.ncols
nrows = table.nrows
# print(ncols, nrows)
table_head = []


def get_avg(j):
    data = []
    for i in range(nrows - 2):
        if (i + 2) % 100 != 0:
            data.append(table.row_values(i + 2)[j])
    # print(data)
    avg = sum(data) / len(data)
    return round(avg, 5)


def get_resnet():
    a = []
    for j in [3, 5, 6, 1]:
        a.append(get_avg(j))
    print(a)


def get_bert():
    a = []
    for j in [3, 5, 6, 7, 8, 9, 10, 11, 12, 1]:
        a.append(get_avg(j))
    print(a)


if ncols == 16:
    get_bert()
if ncols == 10:
    get_resnet()
