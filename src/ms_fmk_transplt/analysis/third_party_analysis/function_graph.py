#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from .function_node import Node


class Graph:
    def __init__(self):
        self.nodelist = {}
        self.numnode = 0

    def addnode(self, key):
        if key not in self.nodelist:
            self.numnode += 1
            new_function_node = Node(key)
            self.nodelist[key] = new_function_node

    def getnode(self, key):
        return self.nodelist.get(key)

    def addedge(self, parent, children):
        if parent not in self.nodelist:
            self.addnode(parent)
        if children not in self.nodelist:
            self.addnode(children)
        self.nodelist[parent].addchildren(self.nodelist[children])

    def get_all_nodes(self):
        return self.nodelist.keys()

    def get_leaf_api(self):
        leaf_apis = []
        for _, node in self.nodelist.items():
            if node.in_degree == 0 and not node.vis:
                leaf_apis.append(node)
        return leaf_apis

    def get_apis(self):
        unsupported_apis = []
        unknown_apis = []
        for _, node in self.nodelist.items():
            if node.has_unsupported_api:
                unsupported_apis.append(node)
            elif node.has_unknown_api:
                unknown_apis.append(node)
        return unsupported_apis, unknown_apis
