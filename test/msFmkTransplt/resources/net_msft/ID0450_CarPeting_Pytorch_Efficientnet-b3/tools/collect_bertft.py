# -*- coding:utf-8 -*-
'''
Created on 2020-05-27

@author: wwx371270
'''
import sys
import commands
import os
import csv
import ascend_function

csvroute = commands.getoutput("pwd") + "/result/"


# 分析性能结果
def analysis_fps(file, batchsize, p):
    cmd = "cat " + str(file) + "| grep \"INFO:tensorflow:loss =\""
    data = commands.getoutput(cmd)
    data = data.split("\n")
    fps_total = 0.0
    num = 0
    for n in range(0, data.__len__()):
        fps = data[n].split(" ")[-2]
        if "(" in fps:
            fps = float(fps.replace("(", ""))
            fps = int(batchsize) * int(p) / fps
            fps_total += fps
            num += 1
    fps_ava = fps_total / num
    return fps_ava


# 分析精度结果
def analysis_accuracy(file):
    cmd = "cat " + str(file) + "| grep \"Loss for final step:\""
    data = commands.getoutput(cmd)
    data = data.split("\n")[-1].split(" ")[-1][:-1]
    loss = float(data)
    return loss


if __name__ == "__main__":
    p = int(sys.argv[5])
    if p == 1:
        trainlog = str(sys.argv[1]) + "/train_0.log"
        batchsize = sys.argv[2]
        resultfile = csvroute + "bert_ft_report.csv"
        if os.path.exists(resultfile) == False:
            with open(resultfile, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(
                    ['train_log', 'train_batch_size', 'max_seq_length', 'predict_batch_size', '1p or 8p',
                     'fps(t_batchsize * p / sec)', 'accuracy(Loss for final step)'])
                f.close()

        cmd = "cat " + trainlog + " |grep \"turing train success\""
        result = commands.getoutput(cmd)
        if "turing train success" in result:
            fps = analysis_fps(trainlog, batchsize, p)
            loss = analysis_accuracy(trainlog)
            if fps / p > 100 and loss < 10:
                row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                    str(fps) + "(success)"] + [str(loss) + "(success)"]
            elif fps / p < 100 and loss < 10:
                row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                    str(fps) + "(fail)"] + [str(loss) + "(success)"]
            elif fps / p > 100 and loss > 10:
                row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                    str(fps) + "(success)"] + [str(loss) + "(fail)"]
            elif fps / p < 100 and loss > 10:
                row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                    str(fps) + "(fail)"] + [str(loss) + "(fail)"]
        else:
            row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + ['NA(fail)'] + [
                'NA(fail)']
        with open(resultfile, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(row)
            f.close()
    else:
        for n in range(0, (p - 1)):
            trainlog = str(sys.argv[1]) + "/train_" + str(n) + ".log"
            batchsize = sys.argv[2]
            resultfile = csvroute + "bert_ft_report.csv"
            if os.path.exists(resultfile) == False:
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(
                        ['train_log', 'train_batch_size', 'max_seq_length', 'predict_batch_size', '1p or 8p',
                         'fps(t_batchsize * p / sec)', 'accuracy(Loss for final step)'])
                    f.close()

            cmd = "cat " + trainlog + " |grep \"turing train success\""
            result = commands.getoutput(cmd)
            if "turing train success" in result:
                fps = analysis_fps(trainlog, batchsize, p)
                loss = analysis_accuracy(trainlog)
                if fps / p > 100 and loss < 10:
                    row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                        str(fps) + "(success)"] + [str(loss) + "(success)"]
                elif fps / p < 100 and loss < 10:
                    row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                        str(fps) + "(fail)"] + [str(loss) + "(success)"]
                elif fps / p > 100 and loss > 10:
                    row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                        str(fps) + "(success)"] + [str(loss) + "(fail)"]
                elif fps / p < 100 and loss > 10:
                    row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                        str(fps) + "(fail)"] + [str(loss) + "(fail)"]
            else:
                row = [trainlog] + [str(batchsize)] + [str(sys.argv[3])] + [str(sys.argv[4])] + [str(p)] + [
                    'NA(fail)'] + [
                          'NA(fail)']
            with open(resultfile, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(row)
                f.close()

