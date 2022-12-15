#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


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


