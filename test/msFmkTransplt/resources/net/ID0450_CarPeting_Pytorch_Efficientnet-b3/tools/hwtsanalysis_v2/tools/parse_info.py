# -*- coding: UTF-8 -*-
import io

"""
@function:
    parse aicore op info.
@author: 
    w00469878, w00282991
@email:
    wangbei5@huawei.com
"""


def get_value(line):
    _, value = line.split(':')
    return value.strip().replace('"', '')


class Tensor(object):
    def __init__(self):
        self.dtype = ''
        self.shape = list()
        self.format = ''
        self.origin_data_type = ''
        self.origin_format = ''
        self.origin_shape = list()
        self.origin_name = ''
        self.origin_output_index = -1

    def dump(self):
        pass


class GraphNode(object):
    def __init__(self):
        self.name = ''
        self.is_fusion_op = False
        self.id = 0
        self.type = ''
        self.block_dim = 0
        self.input = list()
        self.input_desc = list()
        self.output_desc = list()

    def dump(self):
        pass


class Graph(object):
    def __init__(self, proto_file):
        self.nodes = list()
        self.node_map = dict()
        self.parse_info(proto_file)

    def parse_info(self, proto_file):
        lines = []
        with io.open(proto_file, 'r', encoding="gbk") as f:
            for line in f:
                lines.append(line)
        node = None
        tensor = None
        is_input = 0
        is_in_shape = 0
        is_in_origin_shape = 0
        is_in_origin_format = 0
        is_in_origin_data_type = 0
        is_in_origin_name = 0
        is_in_origin_output_index = 0
        is_block_dim = 0
        for line in lines:
            if line.startswith('    name:'):
                if node is not None:
                    self.nodes.append(node)
                    self.node_map[node.name] = node
                node = GraphNode()
                node.name = get_value(line)
            elif line.startswith('      key: "_datadump_original_op_names"'):
                node.is_fusion_op = True
            elif line.startswith('    type:'):
                node.type = get_value(line)
            elif line.startswith('    input:'):
                if line.count(':') == 2:
                    _, value, index = line.split(':')
                    value = value.strip().replace('"', '')
                    index = index.strip().replace('"', '')
                    if index != "-1":
                        node.input.append(value)
            elif line.startswith('    id:'):
                node.id = int(get_value(line))
            elif line.startswith('      key: "tvm_blockdim"'):
                is_block_dim = 1
            elif line.startswith('        i:'):
                if is_block_dim is 1:
                    node.block_dim = int(get_value(line))
                    is_block_dim = 0
            elif line.startswith('    input_desc {'):
                tensor = Tensor()
                is_input = 1
            elif line.startswith('    output_desc {'):
                tensor = Tensor()
                is_input = 0
            elif line.startswith('    }'):
                if tensor is not None:
                    if is_input is 1:
                        node.input_desc.append(tensor)
                        tensor = None
                    else:
                        node.output_desc.append(tensor)
                        tensor = None
            elif line.startswith('      dtype:'):
                tensor.dtype = get_value(line)
            elif line.startswith('      layout:'):
                tensor.format = get_value(line)
            elif line.startswith('      shape {'):
                is_in_shape = 1
            elif line.startswith('      }'):
                if is_in_shape is 1:
                    is_in_shape = 0
            elif line.startswith('        dim:'):
                if is_in_shape is 1:
                    tensor.shape.append(int(get_value(line)))
            elif line.startswith('        key: "origin_format"'):
                is_in_origin_format = 1
            elif line.startswith('        key: "origin_shape"'):
                is_in_origin_shape = 1
            elif line.startswith('        key: "origin_data_type"'):
                is_in_origin_data_type = 1
            elif line.startswith('        key: "_datadump_data_type"'):
                is_in_origin_data_type = 1
            elif line.startswith('        key: "_datadump_origin_format"'):
                is_in_origin_format = 1
            elif line.startswith('        key: "_datadump_origin_name"'):
                is_in_origin_name = 1
            elif line.startswith('        key: "_datadump_origin_output_index"'):
                is_in_origin_output_index = 1
            elif line.startswith('        }'):
                is_in_origin_format = 0
                is_in_origin_shape = 0
                is_in_origin_data_type = 0
                is_in_origin_name = 0
                is_in_origin_output_index = 0
            elif line.startswith('          s:'):
                if is_in_origin_format is 1:
                    tensor.origin_format = get_value(line)
                elif is_in_origin_data_type is 1:
                    if get_value(line) != 'RESERVED':
                        tensor.origin_data_type = get_value(line)
                elif is_in_origin_name is 1:
                    tensor.origin_name = get_value(line)
            elif line.startswith('            i:'):
                if is_in_origin_shape is 1:
                    tensor.origin_shape.append(int(get_value(line)))
            elif line.startswith('          i:'):
                if is_in_origin_output_index is 1:
                    tensor.origin_output_index = int(get_value(line))
        if node is not None:
            self.nodes.append(node)


def parse_info(ge_graph, op_info_file):
    graph = Graph(ge_graph)
    with open(op_info_file, 'w') as op_f:
        for node in graph.nodes:
            node_name = node.name + '_tvmbin'
            if node_name.split('/')[-1] == '_tvmbin':
                node_name = node_name.split('_tvmbin')[0] + 'Layernorm_tvmbin'
            block_dim = node.block_dim
            input_shape = list()
            input_dtype = list()
            input_format = list()
            output_shape = list()
            output_dtype = list()
            output_format = list()
            for tensor in node.input_desc:
                input_shape.append(tensor.shape)
                input_dtype.append(tensor.dtype)
                input_format.append(tensor.format)
            for tensor in node.output_desc:
                output_shape.append(tensor.shape)
                output_dtype.append(tensor.dtype)
                output_format.append(tensor.format)
            op_f.write('%s|%s|%s|%s|%s|%s|%s|%s\n' % (node_name, block_dim, input_shape, input_dtype, input_format,
                                                      output_shape, output_dtype, output_dtype))
