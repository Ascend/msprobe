#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from typing import Optional, Union

import libcst
import libcst.helpers as helper

from utils import transplant_logger as translog
from utils import trans_utils as utils


class ApiVisitor(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider, libcst.metadata.QualifiedNameProvider)

    def __init__(self, op_list):
        super(ApiVisitor, self).__init__()
        self.op_list = op_list
        self.unsupported_op_list = []

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        full_name = self.get_full_name_for_node(node)
        if full_name in self.op_list:
            position = self.get_metadata(libcst.metadata.PositionProvider, node)
            self.unsupported_op_list.append([position.start.line, position.end.line, full_name,
                                             self.op_list.get(full_name)])
        return True

    def print_unsupported_ops(self):
        for unsupported_op in self.unsupported_op_list:
            if unsupported_op[3]:
                unsupported_op_info = "Message: %s" % unsupported_op[3]
            else:
                unsupported_op_info = "Message: %s is not supported now!" % unsupported_op[2]
            msg = "%-21s %-35s %s" % ("line: %s ~ %s" % (unsupported_op[0], unsupported_op[1]),
                                      "Operation Type: UNSUPPORTED", unsupported_op_info)
            translog.warning(msg)
        return self.unsupported_op_list

    def get_full_name_for_node(self, node: Union[str, libcst.CSTNode]) -> Optional[str]:
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        if name_list:
            qualified_name = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))[0].name
        else:
            qualified_name = helper.get_full_name_for_node(node)
        return qualified_name


def get_op_visit_result(code, unsupported_op_list):
    wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
    api_visitor = ApiVisitor(unsupported_op_list)
    module = wrapper.visit(api_visitor)
    op_list = api_visitor.print_unsupported_ops()
    return op_list, module, wrapper
