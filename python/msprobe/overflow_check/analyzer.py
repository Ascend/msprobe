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

import time
from collections import defaultdict
import os
from itertools import dropwhile, chain

from msprobe.core.common.file_utils import check_file_or_directory_path, save_json, make_dir, load_json
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.overflow_check.utils import (
    RankPath,
    FileCache,
    is_communication_op,
    is_ignore_op,
    OverFlowCheckConst,
    analyze_anomaly_in_group,
)
from msprobe.overflow_check.graph import DataNode, CommunicationNode


class OverFlowCheck:
    def __init__(self, input_path, output_path):
        self._input_path = input_path
        self._output_path = output_path
        self._paths = {}
        self._resolve_input_path()
        self._check_framework()
        self._anomaly_nodes = []  # 记录所有异常节点
        self._cache = FileCache()
        self._first_comm_nodes = {}  # 记录各rank下首个通信节点的node_id
        self._after_comm_anomalies = {}  # 记录各rank下发生在通信节点之后的异常计算节点
        self._rank_comm_nodes_dict = {}  # 记录各rank的通信节点

    def analyze(self):
        for analyze_func in [self._pre_analyze, self._analyze, self._post_analyze]:
            analyze_func()
            if self._anomaly_nodes:
                self._gen_analyze_info()
                return
        logger.info('Cannot find any anomaly node, no need to generate analyze file.')

    def _resolve_input_path(self):
        contents = os.listdir(self._input_path)
        for path in contents:
            if len(contents) == 1 and path.startswith('proc'):
                rank = 0
            elif not path.startswith('rank'):
                continue
            else:
                rank_str = path[len('rank') :]
                if not rank_str:
                    rank = 0
                elif not rank_str.isdigit():
                    continue
                else:
                    rank = int(rank_str)
            dump_path = os.path.join(self._input_path, path, OverFlowCheckConst.DUMP_FILE)
            construct_path = os.path.join(self._input_path, path, OverFlowCheckConst.CONSTRUCT_FILE)
            stack_path = os.path.join(self._input_path, path, OverFlowCheckConst.STACK_FILE)
            self._paths[rank] = RankPath(rank, dump_path, construct_path, stack_path)

    def _check_framework(self):
        for _, rankpath in self._paths.items():
            data = load_json(rankpath.dump_path)
            if data.get('framework', '') != "pytorch":
                logger.error(
                    'overflow check is only available for PyTorch, though the data is stored in a different framework.'
                )
                raise RuntimeError(f'invalid dump_path {rankpath.dump_path}')
            break

    def _pre_analyze(self):
        logger.info('Start searching anomaly node before communication.')
        for path in self._paths.values():
            dump_data = self._cache.load_json(path.dump_path).get('data')
            if not dump_data:
                logger.warning(f'Rank {path.rank} has no dump data!')
                continue
            for op_name, op_data in dump_data.items():
                if is_communication_op(op_name):
                    self._first_comm_nodes[path.rank] = op_name
                    break
                data_node = DataNode(op_name, path.rank, op_data)
                if data_node.is_anomaly():
                    self._anomaly_nodes.append(data_node)
                    break

    def _analyze(self):
        logger.info('Start searching anomaly node during communication.')
        self._rank_comm_nodes_dict = {rank: self._analyze_comm_nodes(rank) for rank in self._paths}
        self._connect_comm_nodes()
        self._pruning()
        self._search_first_anomaly()

    def _post_analyze(self):
        logger.info('Start searching anomaly node after communication.')
        for nodes in self._after_comm_anomalies.values():
            if nodes:
                self._anomaly_nodes.append(nodes[0])

    def _gen_analyze_info(self):
        if not os.path.exists(self._output_path):
            make_dir(self._output_path)
        file_name = f'anomaly_analyze_{time.time_ns()}.json'
        result_file = os.path.join(self._output_path, file_name)
        result_content = defaultdict(list)
        for node in self._anomaly_nodes:
            result_content[f'rank_{node.rank}'].append(node.gen_node_info(self._paths[node.rank]))
        save_json(result_file, result_content, 2)
        logger.info(f"The analyze result is saved in: {result_file}")

    def _analyze_comm_nodes(self, rank):
        path = self._paths[rank]
        data = self._cache.load_json(path.dump_path).get('data')
        communication_nodes = {}
        if rank not in self._first_comm_nodes:  # 此rank没有通信节点
            return communication_nodes
        last_node_id = None  # 记录上一个通信节点的node_id
        compute_ops = []  # 记录两个通信节点之间的计算节点
        sub_layer = 0  # 记录两个通信算子之间异常计算节点的调用序数
        for op_name in dropwhile(lambda k: k != self._first_comm_nodes[rank], data):
            node_id = f'{rank}.{op_name}'
            op_data = data[op_name]
            if is_communication_op(op_name):
                comm_node = CommunicationNode(
                    node_id, rank, DataNode(op_name, rank, op_data, sub_layer=sub_layer), compute_ops=compute_ops
                )
                if last_node_id:
                    communication_nodes.get(last_node_id).add_next(comm_node)
                communication_nodes[node_id] = comm_node
                last_node_id = node_id
                compute_ops = []
                sub_layer = 0
            elif not is_ignore_op(op_name):
                data_node = DataNode(op_name, rank, op_data, sub_layer=sub_layer)
                if data_node.is_anomaly():
                    compute_ops.append(data_node)
                sub_layer += 1
        if compute_ops:
            self._after_comm_anomalies[rank] = compute_ops
        return communication_nodes

    def _connect_comm_nodes(self):
        """
        连接跨rank的通信节点，建立关联关系
        """
        comm_index = self._build_comm_nodes_index()
        searched_ranks = set()
        for rank, nodes in list(self._rank_comm_nodes_dict.items())[:-1]:
            searched_ranks.add(rank)
            seen_nodes = set()
            for cur_node in nodes.values():
                conn_info = cur_node.find_connected_nodes()
                if not conn_info.get('ranks'):
                    conn_info['ranks'] = self._rank_comm_nodes_dict.keys()
                if not self._find_connection(conn_info, cur_node, searched_ranks, seen_nodes, comm_index):
                    logger.info(f'Cannot find connected communication node for "{cur_node.node_id}".')

    def _build_comm_nodes_index(self):
        """
        构建通信节点索引：{api: {rank: {call_cnt: node}}}
        时间复杂度：O(R × N)
        """
        comm_index = {}
        for rank, nodes in self._rank_comm_nodes_dict.items():
            for _, node in nodes.items():
                api = node.api
                call_cnt = node.call_cnt
                if api not in comm_index:
                    comm_index[api] = {}
                if rank not in comm_index[api]:
                    comm_index[api][rank] = {}
                comm_index[api][rank][call_cnt] = node
        return comm_index

    def _find_connection(self, conn_info, cur_node, searched_ranks, seen_nodes, comm_index):
        """
        使用索引直接查找连接节点
        时间复杂度：O(1)
        """

        def connect():
            seen_nodes.add(search_node.node_id)
            if search_node.type == OverFlowCheckConst.DST:
                cur_node.add_dst(search_node)
            elif search_node.type == OverFlowCheckConst.SRC:
                search_node.layer = cur_node.layer
                search_node.add_dst(cur_node)
            else:
                cur_node.add_link(search_node)

        found = cur_node.connected
        target_api = conn_info['api'].replace('Distributed.', '')

        if target_api in comm_index:
            for connected_rank in conn_info['ranks']:
                if connected_rank in searched_ranks:
                    continue
                if connected_rank in comm_index[target_api]:
                    target_nodes = comm_index[target_api][connected_rank]
                    for call_cnt, search_node in target_nodes.items():
                        if search_node.node_id in seen_nodes:
                            continue
                        if search_node.type != conn_info.get('type'):
                            continue
                        search_conn_ranks = search_node.find_connected_nodes().get('ranks')
                        if (
                            not search_conn_ranks and search_node.api not in OverFlowCheckConst.DIRECTED_API
                        ) or cur_node.rank in search_conn_ranks:
                            connect()
                            found = True
                            break
        return found

    def _pruning(self):
        deleted_node_id = []
        for nodes in self._rank_comm_nodes_dict.values():
            for node_id in list(nodes.keys()):
                node = nodes[node_id]
                if node.has_nan_inf() or node.compute_ops:
                    continue
                deleted_node_id.append(node_id)
                node.delete()
                del nodes[node_id]
        logger.debug(f'After pruning, following nodes are removed: [{", ".join(deleted_node_id)}]')

    def _search_first_anomaly(self):
        nodes_queues = []
        for comm_nodes in self._rank_comm_nodes_dict.values():
            nodes_queues.append(sorted(list(comm_nodes.values()), key=lambda x: x.layer))
        seen_nodes = set()

        def get_next_node(node_list):
            while node_list:
                next_node = node_list.pop(0)
                if next_node.node_id not in seen_nodes:
                    return next_node
            return None

        def find_all_members(ori_node):
            ids = get_relative_ids(ori_node)
            id_queue = list(chain(*[get_relative_ids(self._get_node_by_id(n_id)).difference(ids) for n_id in ids]))
            while id_queue:
                new_id = id_queue.pop(0)
                ids.add(new_id)
                id_queue.extend(get_relative_ids(self._get_node_by_id(new_id)).difference(ids))
            return ids

        def get_relative_ids(ori_node):
            if not ori_node:
                return set()
            return (
                {ori_node.node_id} | ori_node.link_nodes.keys() | ori_node.src_nodes.keys() | ori_node.dst_nodes.keys()
            )

        while any(nodes_queues):
            groups = []
            all_ids_in_groups = set()
            for nodes in nodes_queues:
                node = get_next_node(nodes)
                if not node:
                    continue
                if not groups or node.node_id in all_ids_in_groups:
                    new_group = find_all_members(node)
                    groups.append(new_group)
                    all_ids_in_groups.update(new_group)
            for group in groups:
                seen_nodes.update(group)
                self._anomaly_nodes.extend(analyze_anomaly_in_group([self._get_node_by_id(n_id) for n_id in group]))
            if self._anomaly_nodes:
                self._anomaly_nodes = [min(self._anomaly_nodes, key=lambda x: (x.layer, x.sub_layer))]
                return

    def _get_node_by_id(self, node_id):
        splits = node_id.split(Const.SEP, 1)
        if len(splits) < 2 or not splits[0].isdigit():
            logger.error(f'invalid node_id {node_id}')
            raise RuntimeError(f'invalid node_id {node_id}')
        rank = int(splits[0])
        return self._rank_comm_nodes_dict.get(rank, {}).get(node_id)


def _overflow_check_parser(parser):
    input_options = ("-i", "--input_path")
    output_options = ("-o", "--output_path")
    parser.add_argument(
        *input_options,
        dest="input_path",
        default="",
        type=str,
        help="<Required> The dump file path, over step level. e.g. \"xxx/step_0/\".",
        required=True,
    )
    parser.add_argument(
        *output_options,
        dest="output_path",
        default="./output",
        type=str,
        help="<optional> The overflow check result output file path.",
        required=False,
    )


def _run_overflow_check(args):
    check_file_or_directory_path(args.input_path, True)
    OverFlowCheck(args.input_path, args.output_path).analyze()
