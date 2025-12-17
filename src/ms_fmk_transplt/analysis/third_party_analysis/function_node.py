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

class Node:
    def __init__(self, key):
        self.key = key
        self.connected_function = set()
        self.has_unsupported_api = False
        self.has_unknown_api = False
        self.vis = False
        self.in_degree = 0
        self.unsupported_list = []
        self.unknown_api_list = []
        self.file_path = ''

    def addchildren(self, children):
        self.connected_function.add(children)

    def get_connections(self):
        return self.connected_function

    def getkey(self):
        return self.key


