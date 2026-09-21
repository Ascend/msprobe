# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from ..utils.graph_utils import GraphUtils
from ..utils.constant import MODULE, NPU, BENCH
from ..utils.global_state import GraphState

from msprobe.visualization.builder.msprobe_adapter import MatchedNodeCalculator


class MatchNodesController:
    @staticmethod
    def is_same_node_type(graph_data, npu_node_name, bench_node_name):
        npu_node_type = graph_data.get("NPU", {}).get("node", {}).get(npu_node_name, {}).get("node_type")
        bench_node_type = graph_data.get("Bench", {}).get("node", {}).get(bench_node_name, {}).get("node_type")
        if npu_node_type is None or bench_node_type is None or npu_node_type != bench_node_type:
            return False
        return True

    @staticmethod
    def process_task_add(graph_data, npu_node_name, bench_node_name, task):
        if not all([graph_data, npu_node_name, bench_node_name, task]):
            return {"success": False, "error": GraphUtils.t("parameterError")}
        if not MatchNodesController.is_same_node_type(graph_data, npu_node_name, bench_node_name):
            return [{"success": False, "error": GraphUtils.t("nodesTypeError")}]
        opposite_npu_node_name = GraphUtils.get_opposite_node_name(npu_node_name)
        opposite_bench_node_name = GraphUtils.get_opposite_node_name(bench_node_name)
        if task == "md5":
            match_result = MatchNodesController.process_md5_task_add(graph_data, npu_node_name, bench_node_name)
            opposite_result = MatchNodesController.process_md5_task_add(
                graph_data, opposite_npu_node_name, opposite_bench_node_name
            )
        elif task == "summary":
            match_result = MatchNodesController.process_summary_task_add(graph_data, npu_node_name, bench_node_name)
            opposite_result = MatchNodesController.process_summary_task_add(
                graph_data, opposite_npu_node_name, opposite_bench_node_name
            )
        elif task == "all":
            match_result = MatchNodesController.process_tensor_task_add(graph_data, npu_node_name, bench_node_name)
            opposite_result = MatchNodesController.process_tensor_task_add(
                graph_data, opposite_npu_node_name, opposite_bench_node_name
            )
        else:
            return [
                {"success": False, "error": GraphUtils.t("taskTypeWrong")},
                {"success": False, "error": GraphUtils.t("taskTypeWrong")},
            ]
        return [match_result, opposite_result]

    @staticmethod
    def process_task_delete(graph_data, npu_node_name, bench_node_name, task):
        opposite_npu_node_name = GraphUtils.get_opposite_node_name(npu_node_name)
        opposite_bench_node_name = GraphUtils.get_opposite_node_name(bench_node_name)
        if task == "md5":
            match_result = MatchNodesController.process_md5_task_delete(graph_data, npu_node_name, bench_node_name)
            opposite_result = MatchNodesController.process_md5_task_delete(
                graph_data, opposite_npu_node_name, opposite_bench_node_name
            )
        elif task == "summary" or task == "all":  # pylint: disable=consider-using-in
            match_result = MatchNodesController.process_summary_task_delete(graph_data, npu_node_name, bench_node_name)
            opposite_result = MatchNodesController.process_summary_task_delete(
                graph_data, opposite_npu_node_name, opposite_bench_node_name
            )
        else:
            return [
                {"success": False, "error": GraphUtils.t("taskTypeWrong")},
                {"success": False, "error": GraphUtils.t("taskTypeWrong")},
            ]
        return [match_result, opposite_result]

    @staticmethod
    def process_task_add_child_layer_by_config(graph_data, match_node_links, task):
        # 根据配置文件中的匹配关系，批量调用 process_task_add
        match_results = []
        for npu_node_name, bench_node_name in match_node_links.items():
            if task == "md5":
                match_result = MatchNodesController.process_md5_task_add(graph_data, npu_node_name, bench_node_name)
            elif task == "summary":
                match_result = MatchNodesController.process_summary_task_add(graph_data, npu_node_name, bench_node_name)
            match_results.append(match_result)  # pylint: disable=possibly-used-before-assignment
        return match_results

    @staticmethod
    def process_task_add_child_layer(graph_data, npu_node_name, bench_node_name, task):
        if not all([graph_data, npu_node_name, bench_node_name, task]):
            return [{"success": False, "error": GraphUtils.t("parameterError")}]

        if not MatchNodesController.is_same_node_type(graph_data, npu_node_name, bench_node_name):
            return [{"success": False, "error": GraphUtils.t("nodesTypeError")}]
        # DB：找到目标节点所有子节点和标杆侧所有子节点
        npu_nodes = graph_data.get("NPU", {}).get("node", {})
        bench_nodes = graph_data.get("Bench", {}).get("node", {})
        result = []
        # 1. 选中的目标节点和标杆侧节点添加匹配关系
        match_result = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)

        # 2. 目标节点的子节点和标杆侧的子节点添加匹配关系

        def process_child_layer(npu_subnodes, bench_subnodes):
            npu_match_names = {}
            bench_match_names = {}
            # 2.1 提取目标侧节点的未匹配子节点名称和标杆侧节点的未匹配子节点名称
            for npu_subnode_name in npu_subnodes:
                node_info = npu_nodes.get(npu_subnode_name, {})
                if node_info.get("matched_node_link"):
                    continue
                node_type = node_info.get("node_type") if node_info else None
                # 2.2 处理节点名称，提取匹配信息
                match_name = (
                    extract_module_name(npu_subnode_name) if node_type == MODULE else extract_api_name(npu_subnode_name)
                )
                matched_name = f"{match_name}.{node_type}"
                npu_match_names.setdefault(matched_name, []).append(npu_subnode_name)

            for bench_subnode_name in bench_subnodes:
                node_info = bench_nodes.get(bench_subnode_name, {})
                if node_info.get("matched_node_link"):
                    continue
                node_type = node_info.get("node_type") if node_info else None
                match_name = (
                    extract_module_name(bench_subnode_name)
                    if node_type == MODULE
                    else extract_api_name(bench_subnode_name)
                )
                matched_name = f"{match_name}.{node_type}"
                bench_match_names.setdefault(matched_name, []).append(bench_subnode_name)

            common_keys = npu_match_names.keys() & bench_match_names.keys()
            # 2.3 取名称交集，添加匹配关系
            for key in common_keys:
                npu_subnode_list = npu_match_names.get(key, [])
                bench_subnode_list = bench_match_names.get(key, [])
                # 多个节点可能有一个module name
                for npu_subnode_name, bench_subnode_name in zip(npu_subnode_list, bench_subnode_list):
                    match_result = MatchNodesController.process_task_add(
                        graph_data, npu_subnode_name, bench_subnode_name, task
                    )
                    npu_subnodes = npu_nodes.get(npu_subnode_name, {}).get("subnodes", [])
                    bench_subnodes = bench_nodes.get(bench_subnode_name, {}).get("subnodes", [])
                    # 2.4 如果有子节点，递归调用2.1-2.4
                    if len(match_result) > 0 and match_result[0].get("success"):
                        result.extend(match_result)
                        if npu_subnodes and bench_subnodes:
                            process_child_layer(npu_subnodes, bench_subnodes)

        def extract_module_name(subnode_name):
            split_subnode_name = subnode_name.split(".")
            if len(split_subnode_name) < 4:
                return ""
            elif not split_subnode_name[-4].isdigit():
                return split_subnode_name[-4]
            elif len(split_subnode_name) > 5:
                return split_subnode_name[-5]
            else:
                return ""

        def extract_api_name(subnode_name):
            split_subnode_name = subnode_name.split(".")
            if len(split_subnode_name) < 2:
                return ""
            elif not split_subnode_name[-2].isdigit():
                return split_subnode_name[-2]
            elif len(split_subnode_name) > 2:
                return split_subnode_name[-3]
            else:
                return ""

        npu_subnodes = npu_nodes.get(npu_node_name, {}).get("subnodes", [])
        bench_subnodes = bench_nodes.get(bench_node_name, {}).get("subnodes", [])
        # 如果当前节点，也就是第一个节点匹配成功，就尝试匹配子节点
        if len(match_result) > 0 and match_result[0].get("success"):
            result.extend(match_result)
            if npu_subnodes and bench_subnodes:
                process_child_layer(npu_subnodes, bench_subnodes)
        return result

    @staticmethod
    def process_task_delete_child_layer(graph_data, npu_node_name, bench_node_name, task):
        if not all([graph_data, npu_node_name, bench_node_name, task]):
            return [{"success": False, "error": GraphUtils.t("parameterError")}]
        result = []
        npu_nodes = graph_data.get("NPU", {}).get("node", {})
        bench_nodes = graph_data.get("Bench", {}).get("node", {})
        # 1. 选中的目标节点和标杆侧节点添加匹配关系
        match_result = MatchNodesController.process_task_delete(graph_data, npu_node_name, bench_node_name, task)

        # 2. 目标节点的子节点和标杆侧的子节点添加匹配关系
        def process_child_layer(npu_child_nodes):
            for npu_subnode_name in npu_child_nodes:
                npu_subnode_info = npu_nodes.get(npu_subnode_name, {})
                matched_node_link = npu_subnode_info.get("matched_node_link", [])
                if not matched_node_link:
                    continue
                bench_subnode_name = matched_node_link[-1]
                match_result = MatchNodesController.process_task_delete(
                    graph_data, npu_subnode_name, bench_subnode_name, task
                )
                npu_subnodes = npu_nodes.get(npu_subnode_name, {}).get("subnodes", [])
                bench_subnodes = bench_nodes.get(bench_subnode_name, {}).get("subnodes", [])
                # 2.4 如果有子节点，递归调用2.1-2.4
                if len(match_result) > 0 and match_result[0].get("success"):
                    result.extend(match_result)
                    if npu_subnodes and bench_subnodes:
                        process_child_layer(npu_subnodes)

        npu_subnodes = npu_nodes.get(npu_node_name, {}).get("subnodes", [])
        bench_subnodes = bench_nodes.get(bench_node_name, {}).get("subnodes", [])
        # 如果当前节点，也就是第一个节点匹配成功，就尝试匹配子节点
        if len(match_result) > 0 and match_result[0].get("success"):
            result.extend(match_result)
            if npu_subnodes and bench_subnodes:
                process_child_layer(npu_subnodes)
        return result

    @staticmethod
    def process_md5_task_add(graph_data, npu_node_name, bench_node_name):
        npu_node_data = graph_data.get("NPU", {}).get("node", {}).get(npu_node_name)
        bench_node_data = graph_data.get("Bench", {}).get("node", {}).get(bench_node_name)
        if not npu_node_data or not bench_node_data:
            return {
                "success": False,
                "error": f'{npu_node_name}{GraphUtils.t("or")}{bench_node_name}{GraphUtils.t("nodeNotExist")}',
            }

        # 计算精度误差
        npu_data = {
            "input_data": npu_node_data.get("input_data", {}),
            "output_data": npu_node_data.get("output_data", {}),
        }
        bench_data = {
            "input_data": bench_node_data.get("input_data", {}),
            "output_data": bench_node_data.get("output_data", {}),
        }
        m = MatchedNodeCalculator(npu_data, bench_data)
        result = m.get_db_md5_compare_result()
        # 在原始数据上，添加匹配节点，和匹配节点信息
        # JSON：处理JSON更新
        npu_graph_data = graph_data.get("NPU", {})
        bench_graph_data = graph_data.get("Bench", {})
        npu_node_data["matched_node_link"] = GraphUtils.get_parent_node_list(bench_graph_data, bench_node_name)
        bench_node_data["matched_node_link"] = GraphUtils.get_parent_node_list(npu_graph_data, npu_node_name)
        npu_node_data.setdefault("data", {})["precision_index"] = result.get("data", {}).get("precision_index")
        # DB: data只有DB会用到
        data = [
            {
                "node_name": npu_node_name,
                "matched_node_link": npu_node_data["matched_node_link"],
                "precision_index": result.get("data", {}).get("precision_index"),
                "input_data": npu_node_data.get("input_data"),
                "output_data": npu_node_data.get("output_data"),
                "graph_type": NPU,
            },
            {
                "node_name": bench_node_name,
                "matched_node_link": bench_node_data["matched_node_link"],
                "precision_index": None,
                "input_data": bench_node_data.get("input_data"),
                "output_data": bench_node_data.get("output_data"),
                "graph_type": BENCH,
            },
        ]
        MatchNodesController.add_config_match_nodes(npu_node_name, bench_node_name)
        return {"success": True, "data": data}

    @staticmethod
    def process_summary_task_add(graph_data, npu_node_name, bench_node_name):
        # 节点信息提取
        npu_node_data = graph_data.get("NPU", {}).get("node", {}).get(npu_node_name, {})
        bench_node_data = graph_data.get("Bench", {}).get("node", {}).get(bench_node_name, {})
        # 计算统计误差
        npu_data = {
            "input_data": npu_node_data.get("input_data", {}),
            "output_data": npu_node_data.get("output_data", {}),
        }
        bench_data = {
            "input_data": bench_node_data.get("input_data", {}),
            "output_data": bench_node_data.get("output_data", {}),
        }
        m = MatchedNodeCalculator(npu_data, bench_data)
        result = m.get_db_statistics_compare_result()
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "")}  # 如果没有匹配成功，直接返回错误信息
        # 在原始数据上，添加匹配节点，和匹配节点信息
        # JSON：处理JSON更新
        npu_graph_data = graph_data.get("NPU", {})
        bench_graph_data = graph_data.get("Bench", {})
        npu_node_data["matched_node_link"] = GraphUtils.get_parent_node_list(bench_graph_data, bench_node_name)
        bench_node_data["matched_node_link"] = GraphUtils.get_parent_node_list(npu_graph_data, npu_node_name)
        npu_node_data.setdefault("data", {})["precision_index"] = result.get("data", {}).get("precision_index")
        MatchNodesController.update_graph_node_data(
            npu_node_data.get("input_data"), result.get("data", {}).get("input_info", {}), "summary"
        )
        MatchNodesController.update_graph_node_data(
            npu_node_data.get("output_data"), result.get("data", {}).get("output_info", {}), "summary"
        )
        # DB: data只有DB会用到
        data = [
            {
                "node_name": npu_node_name,
                "matched_node_link": npu_node_data.get("matched_node_link"),
                "precision_index": result.get("data", {}).get("precision_index"),
                "input_data": npu_node_data.get("input_data"),
                "output_data": npu_node_data.get("output_data"),
                "graph_type": NPU,
            },
            {
                "node_name": bench_node_name,
                "matched_node_link": bench_node_data.get("matched_node_link"),
                "precision_index": None,
                "input_data": bench_node_data.get("input_data"),
                "output_data": bench_node_data.get("output_data"),
                "graph_type": BENCH,
            },
        ]
        MatchNodesController.add_config_match_nodes(npu_node_name, bench_node_name)

        return {"success": True, "data": data}

    ## Tensor 任务的手动匹配策略
    @staticmethod
    def process_tensor_task_add(graph_data, npu_node_name, bench_node_name):
        # 节点信息提取
        npu_node_data = graph_data.get("NPU", {}).get("node", {}).get(npu_node_name, {})
        bench_node_data = graph_data.get("Bench", {}).get("node", {}).get(bench_node_name, {})
        # 计算统计误差
        npu_data = {
            "input_data": npu_node_data.get("input_data", {}),
            "output_data": npu_node_data.get("output_data", {}),
            "dump_data_dir": npu_node_data.get("dump_data_dir", ""),
        }
        bench_data = {
            "input_data": bench_node_data.get("input_data", {}),
            "output_data": bench_node_data.get("output_data", {}),
            "dump_data_dir": bench_node_data.get("dump_data_dir", ""),
        }
        m = MatchedNodeCalculator(npu_data, bench_data)
        result = m.get_db_tensor_compare_result()
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "")}
        # 在原始数据上，添加匹配节点，和匹配节点信息
        # JSON：处理JSON更新
        npu_graph_data = graph_data.get("NPU", {})
        bench_graph_data = graph_data.get("Bench", {})
        npu_node_data["matched_node_link"] = GraphUtils.get_parent_node_list(bench_graph_data, bench_node_name)
        bench_node_data["matched_node_link"] = GraphUtils.get_parent_node_list(npu_graph_data, npu_node_name)
        npu_node_data.setdefault("data", {})["precision_index"] = result.get("data", {}).get("precision_index")
        MatchNodesController.update_graph_node_data(
            npu_node_data.get("input_data"), result.get("data", {}).get("input_info", {}), "all"
        )
        MatchNodesController.update_graph_node_data(
            npu_node_data.get("output_data"), result.get("data", {}).get("output_info", {}), "all"
        )
        # DB: data只有DB会用到
        data = [
            {
                "node_name": npu_node_name,
                "matched_node_link": npu_node_data.get("matched_node_link"),
                "precision_index": result.get("data", {}).get("precision_index"),
                "input_data": npu_node_data.get("input_data"),
                "output_data": npu_node_data.get("output_data"),
                "graph_type": NPU,
            },
            {
                "node_name": bench_node_name,
                "matched_node_link": bench_node_data.get("matched_node_link"),
                "precision_index": None,
                "input_data": bench_node_data.get("input_data"),
                "output_data": bench_node_data.get("output_data"),
                "graph_type": BENCH,
            },
        ]
        MatchNodesController.add_config_match_nodes(npu_node_name, bench_node_name)
        return {"success": True, "data": data}

    @staticmethod
    def process_md5_task_delete(graph_data, npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        npu_match_nodes_list = config_data.get("npuMatchNodes", {})
        bench_match_nodes_list = config_data.get("benchMatchNodes", {})
        if (
            npu_match_nodes_list.get(npu_node_name) != bench_node_name
            or bench_match_nodes_list.get(bench_node_name) != npu_node_name
        ):
            return {
                "success": False,
                "error": f"{GraphUtils.t('operationFailed')}{npu_node_name}{GraphUtils.t('or')}{bench_node_name}{GraphUtils.t('nodeUnmatchError')}",
            }
        npu_node_data = graph_data.get("NPU", {}).get("node", {}).get(npu_node_name)
        bench_node_data = graph_data.get("Bench", {}).get("node", {}).get(bench_node_name)
        if not npu_node_data or not bench_node_data:
            return {
                "success": False,
                "error": f"{GraphUtils.t('operationFailed')}{npu_node_name}{GraphUtils.t('or')}{bench_node_name}{GraphUtils.t('nodeNotExist')}",
            }
        # 在原始数据上，删除匹配节点，和匹配节点信息
        # JSON：处理JSON更新
        npu_node_data["matched_node_link"] = []
        bench_node_data["matched_node_link"] = []
        del npu_node_data["data"]["precision_index"]
        # DB: data只有DB会用到
        data = [
            {
                "node_name": npu_node_name,
                "matched_node_link": [],
                "precision_index": None,
                "input_data": npu_node_data.get("input_data"),
                "output_data": npu_node_data.get("output_data"),
                "graph_type": NPU,
            },
            {
                "node_name": bench_node_name,
                "matched_node_link": [],
                "precision_index": None,
                "input_data": bench_node_data.get("input_data"),
                "output_data": bench_node_data.get("output_data"),
                "graph_type": BENCH,
            },
        ]
        MatchNodesController.delete_config_match_nodes(npu_node_name, bench_node_name)
        return {
            "success": True,
            "data": data,
        }

    @staticmethod
    def process_summary_task_delete(graph_data, npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        npu_match_nodes_list = config_data.get("npuMatchNodes", {})
        bench_match_nodes_list = config_data.get("benchMatchNodes", {})
        if (
            npu_match_nodes_list.get(npu_node_name) != bench_node_name
            or bench_match_nodes_list.get(bench_node_name) != npu_node_name
        ):
            return {
                "success": False,
                "error": f"{GraphUtils.t('operationFailed')}{npu_node_name, bench_node_name}{GraphUtils.t('nodeUnmatchError')}",
            }
        npu_node_data = graph_data.get("NPU", {}).get("node", {}).get(npu_node_name)
        bench_node_data = graph_data.get("Bench", {}).get("node", {}).get(bench_node_name)
        if not npu_node_data or not bench_node_data:
            return {
                "success": False,
                "error": f"{GraphUtils.t('operationFailed')}{npu_node_name}{GraphUtils.t('or')}{bench_node_name}{GraphUtils.t('nodeNotExist')}",
            }
        # DB：更新数据的节点信息，直接返回新的结果，此处暂时不做更新
        # JSON：处理JSON更新
        npu_node_data["matched_node_link"] = []
        bench_node_data["matched_node_link"] = []
        input_data = MatchNodesController.delete_matched_node_data(npu_node_data.get("input_data"))
        output_data = MatchNodesController.delete_matched_node_data(npu_node_data.get("output_data"))
        # 防止 KeyError {GraphUtils.t('or')} TypeError
        npu_node_data.get("data", {}).pop("precision_index", None)
        # DB: data只有DB会用到
        data = [
            {
                "node_name": npu_node_name,
                "matched_node_link": [],
                "precision_index": None,
                "input_data": input_data,
                "output_data": output_data,
                "graph_type": NPU,
            },
            {
                "node_name": bench_node_name,
                "matched_node_link": [],
                "precision_index": None,
                "input_data": bench_node_data.get("input_data"),
                "output_data": bench_node_data.get("output_data"),
                "graph_type": BENCH,
            },
        ]
        MatchNodesController.delete_config_match_nodes(npu_node_name, bench_node_name)
        return {"success": True, "data": data}

    @staticmethod
    def add_config_match_nodes(npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        # 匹配列表和未匹配列表
        manual_match_nodes = config_data.setdefault("manualMatchNodes", {})
        npu_match_nodes_list = config_data.setdefault("npuMatchNodes", {})
        bench_match_nodes_list = config_data.setdefault("benchMatchNodes", {})
        npu_unmatehed_name_list = config_data.setdefault("npuUnMatchNodes", [])
        bench_unmatehed_name_list = config_data.setdefault("benchUnMatchNodes", [])
        # 更新匹配列表和未匹配列表
        if str(npu_node_name) in npu_unmatehed_name_list:
            npu_unmatehed_name_list.remove(str(npu_node_name))
        if str(bench_node_name) in bench_unmatehed_name_list:
            bench_unmatehed_name_list.remove(str(bench_node_name))
        manual_match_nodes[str(npu_node_name)] = str(bench_node_name)
        npu_match_nodes_list[str(npu_node_name)] = str(bench_node_name)
        bench_match_nodes_list[str(bench_node_name)] = str(npu_node_name)
        GraphState.set_global_value("config_data", config_data)

    @staticmethod
    def delete_config_match_nodes(npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        # 匹配列表和未匹配列表
        manual_match_nodes = config_data.setdefault("manualMatchNodes", {})
        npu_match_nodes_list = config_data.setdefault("npuMatchNodes", {})
        bench_match_nodes_list = config_data.setdefault("benchMatchNodes", {})
        npu_unmatehed_name_list = config_data.setdefault("npuUnMatchNodes", [])
        bench_unmatehed_name_list = config_data.setdefault("benchUnMatchNodes", [])
        # 更新匹配列表和未匹配列表
        if str(npu_node_name) in manual_match_nodes:
            del manual_match_nodes[str(npu_node_name)]
        if str(npu_node_name) in npu_match_nodes_list:
            del npu_match_nodes_list[str(npu_node_name)]
        if str(bench_node_name) in bench_match_nodes_list:
            del bench_match_nodes_list[str(bench_node_name)]
        npu_unmatehed_name_list.append(str(npu_node_name))
        bench_unmatehed_name_list.append(str(bench_node_name))
        GraphState.set_global_value("config_data", config_data)

    @staticmethod
    def update_graph_node_data(graph_npu_node_data, statistical_diff, task):
        if not statistical_diff or not graph_npu_node_data:
            return
        for key, diff_values in statistical_diff.items():
            if task == "summary":
                # 格式化相对误差字段
                for field in ["MaxRelativeErr", "MinRelativeErr", "NormRelativeErr", "MeanRelativeErr"]:
                    diff_values[field] = diff_values.get(field, float("nan"))
                # 转换 absErr 为 NaN 字符串
                for field in MatchedNodeCalculator.STATISTICS_COMPARE_INDEX:
                    diff_values[field] = diff_values.get(field, float("nan"))
            if task == "all":
                for filed in MatchedNodeCalculator.TENSOR_COMPARE_INDEX:
                    diff_values[filed] = diff_values.get(filed, float("nan"))

            if key in graph_npu_node_data:
                graph_npu_node_data[key].update(diff_values)
            else:
                graph_npu_node_data[key] = diff_values

    @staticmethod
    def delete_matched_node_data(graph_npu_node_data):
        # 遍历graph_npu_node_data中的每个主键和对应的子字典
        for key, fild_obj in graph_npu_node_data.items():
            if not fild_obj or not isinstance(fild_obj, dict):
                continue
            # 使用字典解析创建新的子字典，排除不需要的键
            graph_npu_node_data[key] = {
                sub_key: value
                for sub_key, value in fild_obj.items()
                if sub_key not in MatchedNodeCalculator.TENSOR_COMPARE_INDEX
                and sub_key not in MatchedNodeCalculator.STATISTICS_COMPARE_INDEX
            }
        return graph_npu_node_data
