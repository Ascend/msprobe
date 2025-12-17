#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
import sys
import os

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))


class TestModelArtsPathManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.ms_fmk_transplt.transfer.adapter.ascend_modelarts_function import ModelArtsPathManager
        cls.ModelArtsPathManager = ModelArtsPathManager

    def test_get_path_not_on_modelarts(self):
        manager = self.ModelArtsPathManager()
        manager._is_run_on_modelarts = False
        manager.project_path = '/workspace/project'
        assert manager.get_path('./data') == './data'
        assert manager.get_path('/data/dataset') == '/data/dataset'
        assert manager.get_path(root='./data') == './data'

    def test_get_input_path_of_directory_on_modelarts(self):
        manager = self.ModelArtsPathManager()
        manager._is_run_on_modelarts = True
        manager.project_path = '/workspace/project'
        manager._input_path_mapping = {
            '/home/x2mindspore/dataset/imagenet': self.ModelArtsPathManager.PathPair(self.ModelArtsPathManager.PathType.DIR,
                                                                                '/modelarts/dataset/imagenet',
                                                                                's3://dataset/imagenet')
        }
        manager._output_path_mapping = {}

        assert manager.get_path('./data') == '/workspace/project/data'
        assert manager.get_path('data') == '/workspace/project/data'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet') == '/modelarts/dataset/imagenet'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet/') == '/modelarts/dataset/imagenet'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet/train') == '/modelarts/dataset/imagenet/train'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet/train/') == '/modelarts/dataset/imagenet/train'
        assert manager.get_path('/home/x2mindspore/dataset2/imagenet') == '/home/x2mindspore/dataset2/imagenet'

    def test_get_input_path_of_file_on_modelarts(self):
        manager = self.ModelArtsPathManager()
        manager._is_run_on_modelarts = True
        manager.project_path = '/workspace/project'
        manager._input_path_mapping = {
            '/home/x2mindspore/dataset/imagenet/classes.txt':
                self.ModelArtsPathManager.PathPair(self.ModelArtsPathManager.PathType.FILE,
                                              '/modelarts/dataset/imagenet/classes.txt',
                                              's3://dataset/imagenet/classes.txt')
        }
        manager._output_path_mapping = {}

        assert manager.get_path('./data') == '/workspace/project/data'
        assert manager.get_path('data') == '/workspace/project/data'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet') == '/home/x2mindspore/dataset/imagenet'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet/classes.txt') == \
               '/modelarts/dataset/imagenet/classes.txt'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet/classes.txt/') == \
               '/home/x2mindspore/dataset/imagenet/classes.txt/'
        assert manager.get_path('/home/x2mindspore/dataset/imagenet/classes.txt/sub_path') == \
               '/home/x2mindspore/dataset/imagenet/classes.txt/sub_path'

    def test_get_output_path_of_directory_on_modelarts(self):
        manager = self.ModelArtsPathManager()
        manager._is_run_on_modelarts = True
        manager.project_path = '/workspace/project'
        manager._input_path_mapping = {}
        manager._output_path_mapping = {
            './output': self.ModelArtsPathManager.PathPair(
                self.ModelArtsPathManager.PathType.DIR, '/workspace/project/output', 's3://output/'),
            '/home/x2mindspore/checkpoints': self.ModelArtsPathManager.PathPair(
                self.ModelArtsPathManager.PathType.DIR, '/modelarts/checkpoints', 's3://checkpoints'),
        }

        assert manager.get_path('./output') == '/workspace/project/output'
        assert manager.get_path('./output/') == '/workspace/project/output'
        assert manager.get_path('output') == '/workspace/project/output'
        assert manager.get_path('output/logs') == '/workspace/project/output/logs'

        assert manager.get_path('/home/x2mindspore/checkpoints') == '/modelarts/checkpoints'
        assert manager.get_path('/home/x2mindspore/checkpoints/') == '/modelarts/checkpoints'
        assert manager.get_path('/home/x2mindspore/checkpoints/best.ckpt') == \
               '/modelarts/checkpoints/best.ckpt'
        assert manager.get_path('/home/x2mindspore/output2/checkpoints') == '/home/x2mindspore/output2/checkpoints'

    def test_get_output_path_of_file_on_modelarts(self):
        manager = self.ModelArtsPathManager()
        manager._is_run_on_modelarts = True
        manager.project_path = '/workspace/project'
        manager._input_path_mapping = {}
        manager._output_path_mapping = {
            './checkpoint/best.ckpt': self.ModelArtsPathManager.PathPair(
                self.ModelArtsPathManager.PathType.FILE, '/workspace/project/checkpoint/best.ckpt',
                's3://checkpoint/best.ckpt'),
            '/home/x2mindspore/log/0.log': self.ModelArtsPathManager.PathPair(
                self.ModelArtsPathManager.PathType.FILE, '/modelarts/log/0.log', 's3://log/0.log')
        }

        assert manager.get_path('./checkpoint/best.ckpt') == '/workspace/project/checkpoint/best.ckpt'
        assert manager.get_path('/home/x2mindspore/log/0.log') == '/modelarts/log/0.log'
        assert manager.get_path('/home/x2mindspore/log/0.log/') == '/home/x2mindspore/log/0.log/'


if __name__ == '__main__':
    unittest.main()
