#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import datetime
import json
import os
import shutil
import sys
import unittest
import unittest.mock as mock
import difflib
import io
from multiprocessing import Process
from multiprocessing import Manager

import xmlrunner
from xmlrunner.extra.xunit_plugin import transform
from test_rules import TestRules as TestBuildRules

import coverage

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/msFmkTransplt"))

TRANS_ERROR=1

class Args(object):
    def __init__(self, input_path, output_path, main=None, target_model='model'):
        self.input = input_path
        self.output = output_path
        self.rule = ''
        self.specify_device = False
        self.device_id = 0
        self.similar = True
        if main:
            self.main = main
            self.target_model = target_model


def run(mock_args, net_name, result_dict, output_path):
    from src.msFmkTransplt.ms_fmk_transplt import MsFmkTransplt
    try:
        ms_fmk_transplt = MsFmkTransplt()
        ms_fmk_transplt._MsFmkTransplt__parse_command = mock_args
        ret = ms_fmk_transplt.main()
        if output_path is not None:
            shutil.rmtree(output_path + "/" + net_name + '_msft/ascend_function')
        result_dict[net_name] = 0 if ret == 0 else TRANS_ERROR
    except Exception as e:
        print(repr(e))
        result_dict[net_name] = TRANS_ERROR

class TestMsFmkTransplt(unittest.TestCase):

    def setUp(self):
        import src.msFmkTransplt.ms_fmk_transplt
        self.abs_input_path = os.path.abspath('../resources/net')
        shutil.rmtree("../test_result/", ignore_errors=True)
        os.makedirs("../test_result/net_msft", exist_ok=True)
        self.abs_output_path = os.path.abspath("../test_result") + "/net_msft"
        self.standard_dir = os.path.abspath("../resources/net_msft")
        self.log_file_name = "msFmkTranspltlog.txt"
        self.input_py_file_list = []
        self.output_py_file_list = []
        self.standard_py_file_list = []
        self.list_python_file(self.abs_input_path)
        self.has_error = False


    def list_python_file(self, path):
        files = os.listdir(path)
        for file_name in files:
            sub_file = path + '/' + file_name
            if os.path.isdir(sub_file) and os.path.basename(sub_file) != 'ascend_function':
                self.list_python_file(sub_file)
            elif os.path.isfile(sub_file) and sub_file.endswith(".py"):
                self.input_py_file_list.append(sub_file)
                self.output_py_file_list.append(sub_file.replace(self.abs_input_path, self.abs_output_path))
                self.standard_py_file_list.append(sub_file.replace(self.abs_input_path, self.standard_dir))

    def test_main(self):
        result_dict = transplt(self.abs_input_path, self.abs_output_path)

        self.assertFalse(TRANS_ERROR in result_dict.values())

        result_dict = transplt_multi(self.abs_input_path, self.abs_output_path)

        self.assertFalse(TRANS_ERROR in result_dict.values())

        print("-----------------Begin to compare result---------------------")

        for i in range(len(self.standard_py_file_list)):
            standard_file = self.standard_py_file_list[i]
            output_file = self.output_py_file_list[i]
            self.result_check(standard_file, output_file)
        self.assertFalse(self.has_error)

    def result_check(self, standard_file, output_file):
        with open(standard_file, 'r', encoding='utf-8') as st_file:
            standard_content = st_file.read().splitlines()
        with open(output_file, 'r', encoding='utf-8') as out_file:
            output_content = out_file.read().splitlines()
        result = list(difflib.unified_diff(standard_content, output_content, n=0))
        if result:
            print('\n\n-------------------------------------------------------------------------', flush=True)
            print(f'[Error] {output_file.replace(self.abs_output_path, "")} conversion results are inconsistent.',
                  flush=True)
            print('\n'.join(result), flush=True)
            print('-------------------------------------------------------------------------', flush=True)
            self.has_error = True

    def read_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                content = file.readlines()
        except FileNotFoundError:
            self.fail("File not exist error!")
        except PermissionError:
            self.fail("Read file permission error!")
        return content

    def load_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                content = json.load(file)
        except FileNotFoundError:
            self.fail("Json file not exist error!")
        except PermissionError:
            self.fail("Load json permission error!")
        return content.get("reports")


def transplt(input_path, output_path, standard_dir=None):
    process_list = []
    result_dict = Manager().dict()
    for file in os.listdir(input_path):
        if file.endswith("_multi"):
            continue
        mock_args = mock.Mock(return_value=Args(input_path + '/' + file, output_path))
        process = Process(target=run, args=(mock_args, file, result_dict, standard_dir))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    for file in os.listdir(input_path):
        if file.endswith("_multi"):
            continue
        os.rename(output_path + '/' + file + '_msft', output_path + '/' + file)
    for key, value in result_dict.items():
        if value != 0:
            print(f"{key} translates failed.")
    return result_dict


def transplt_multi(input_path, output_path, standard_dir=None):
    process_list = []
    result_dict = Manager().dict()
    main_file_dict = {
        'ID0339_CarPeting_Pytorch_EAST_multi': input_path + '/ID0339_CarPeting_Pytorch_EAST_multi/train_ICDAR15.py',
        'ID0476_CarPeting_Pytorch_3D_nested_unet_multi': input_path + '/ID0476_CarPeting_Pytorch_3D_nested_unet_multi/train.py',
        'ID0478_CarPeting_Pytorch_3D_attentionnet_multi': input_path + '/ID0478_CarPeting_Pytorch_3D_attentionnet_multi/train.py',
        'ID0669_CarPeting_Pytorch_GENet_multi': input_path + '/ID0669_CarPeting_Pytorch_GENet_multi/train.py',
    }
    for file, main_file in main_file_dict.items():
        mock_args = mock.Mock(return_value=Args(input_path + '/' + file, output_path, main_file))
        process = Process(target=run, args=(mock_args, file, result_dict, standard_dir))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    for file in main_file_dict.keys():
        os.rename(output_path + '/' + file + '_msft', output_path + '/' + file)
    for key, value in result_dict.items():
        if value != 0:
            print(f"{key} multi translates failed.")
    return result_dict


def update_standard():
    abs_input_path = os.path.abspath('../resources/net')
    standard_dir = os.path.abspath("../resources/net_msft")
    shutil.rmtree(standard_dir, ignore_errors=True)
    os.makedirs(standard_dir)

    transplt(abs_input_path, standard_dir, standard_dir)
    transplt_multi(abs_input_path, standard_dir, standard_dir)

    print("Standard file update finished.")

    with open('../resources/updateLog.txt', 'a+') as f:
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{now_time}, name, issue/requirement/DTS, reason\n")
    print("The update time has been written into test/msFmkTransplt/resources/updateLog.txt, "
          "please continue to add the name and reason for modification in it.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        update_standard()
    else:
        src_list = ["src.msFmkTransplt"]
        cov = coverage.Coverage(concurrency="multiprocessing", source=src_list, cover_pylib=False,
                                     omit=["*/libcst/*", "test*", "*xmlrunner*", "*site-packages*"], branch=True)
        if len(sys.argv) > 1 and sys.argv[1] == 'mr':
            del sys.argv[1]
            out = io.BytesIO()
            runner = xmlrunner.XMLTestRunner(output=out)
            cov.start()
            unittest.main(testRunner=runner, exit=False)
            cov.stop()
            with open('./final.xml', 'wb') as report:
                report.write(transform(out.getvalue()))
            cov.save()
            cov.combine()
            cov.report()
            cov.xml_report(outfile="./coverage.xml")
        else:
            cov.start()
            unittest.main(exit=False)
            cov.stop()
            cov.save()
            cov.combine()
            cov.report()
            cov.html_report(directory="./report")
