# -*- coding:utf-8 -*-
import commands
import time
import sys
import os


def cleandump():
    cmd1 = "find /var/log/npu/ide_daemon/dump/tmp -name \"ReduceMeanD*\""
    dump_route = commands.getoutput(cmd1).split("\n")
    print(dump_route)
    model_route = []
    if dump_route[0] != "":
        for n in range(0, dump_route.__len__()):
            remove_route = dump_route[n].split('/')[-2] + "/" + dump_route[n].split('/')[-1]
            model_route.append(dump_route[n].replace(remove_route, ""))

        model_route = list(set(model_route))

        for n in range(0, model_route.__len__()):
            cmd2 = "ls -lrt " + model_route[n] + " | awk '{print $9}'"
            index_route = commands.getoutput(cmd2).split("\n")
            if index_route[0] == "":
                index_route.pop(0)
            if index_route.__len__() > 3:
                for m in range(0, index_route.__len__() - 3):
                    cmd = "cd " + model_route[n] + " && rm -rf " + index_route[m]
                    commands.getoutput(cmd)


if __name__ == "__main__":
    while 1:
        cleandump()
        time.sleep(60)
