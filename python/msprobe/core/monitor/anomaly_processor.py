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
# pylint: disable=duplicate-code

import os
import sys
import math
import ast
import heapq
from abc import ABC
from dataclasses import dataclass, field
from typing import List

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.log import logger
from msprobe.core.common.cli_help import MindStudioArgumentParser
from msprobe.core.common.file_utils import (
    save_json,
    create_directory,
    remove_path,
    check_file_or_directory_path,
    load_json,
)


class ScanRule(ABC):
    name = "ScanRule"

    def apply(self, cur, history=None):
        raise NotImplementedError("abstract method apply is not implemented")


class AnomalyTurbulence(ScanRule):
    name = "AnomalyTurbulence"

    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def apply(self, cur, history=None):
        """
        :param cur: float, current metric value
        :param history: float, history weighted average
        :return: bool, whether the current value deviates from the historical average value of current metric
        """
        up_bound = history * (1 + self.threshold)
        return abs(cur) > up_bound


class AnomalyNan(ScanRule):
    name = "AnomalyNan"

    def __init__(self, threshold=None) -> None:
        self.threshold = threshold

    def apply(self, cur, history=None):
        return math.isnan(cur) or (self.threshold is not None and abs(cur) > self.threshold)


class AnomalyScanner:
    @staticmethod
    def load_rules(specs: List[dict]):
        """
        specs: [{"rule_name": "AnomalyTurbulence", "args": {"threshold": 0.5}}]
        """
        if specs is None:
            return []
        alert_rules = []
        for spec in specs:
            # 使用get方法获取键值，如果键不存在则返回None
            rule_cls_name = spec.get("rule_name")
            rule_args = spec.get("args")

            # 检查必要的键是否存在
            if rule_cls_name is None or (rule_cls_name == "AnomalyTurbulence" and rule_args is None):
                logger.warning(f"Spec is missing required keys: {spec}")
                continue

            cur_module = sys.modules.get(__name__)
            try:
                rule_cls = getattr(cur_module, rule_cls_name)
            except AttributeError:
                logger.error(f"Rule class '{rule_cls_name}' not found in the current module.")
                continue

            try:
                rule_instance = rule_cls(**rule_args) if rule_args is not None else rule_cls()
                alert_rules.append(rule_instance)
            except Exception as e:
                logger.error(f"Error creating instance of rule '{rule_cls_name}': {e}")
                continue

        return alert_rules

    @staticmethod
    def scan(scan_rules: List[ScanRule], history, cur):
        anomaly = False
        for rule in scan_rules:
            anomaly = rule.apply(cur, history=history)
            if anomaly:
                return anomaly, rule.name
        return anomaly, None


class AnomalyDataFactory(ABC):
    def __init__(self, rank, pp_stage, group_mates):
        super().__init__()
        self.rank = rank
        self.pp_stage = pp_stage
        self.group_mates = group_mates
        self.micro_step = 0
        self.name2callid = {}

    def set_call_id(self, name2callid):
        """根据当前GradContext信息更新call_id vpp_stage等信息"""
        self.name2callid = name2callid

    def create(self, tag, message, step):
        """如果检查出异常, 调用当前接口生成GradAnomalyData实例
        tag (tuple): metric tag ('0:1.post_attention_norm.weight/rank0/pre_grad', 'min')
        message (str): anomaly detect message
        step (int): training step
        """
        if not isinstance(tag, tuple) or len(tag) != 2:
            raise ValueError("tag must be a tuple with length 2")
        tag_name = tag[0]
        param_name = tag_name.split('/')[0]
        call_id = self.name2callid.get(tag_name, -1)
        if MonitorConst.NAME_SEP in param_name:
            vpp_stage = int(param_name.split(MonitorConst.NAME_SEP)[0])
        else:
            vpp_stage = 0

        return GradAnomalyData(
            self.rank, step, self.micro_step, self.pp_stage, vpp_stage, call_id, tag_name, message, self.group_mates
        )


@dataclass(eq=True)
class GradAnomalyData:
    rank: int = 0
    step: int = 0
    micro_step: int = 0
    pp_stage: int = 0
    vpp_stage: int = 0
    call_id: int = 0
    tag_name: str = field(default=None, compare=False)
    message: str = field(default="", compare=False)
    group_mates: list = field(default=None, compare=False)

    def __lt__(self, other):
        """
        自定义比较函数，用于确定 GradAnomalyData 实例之间的顺序。
        比较规则为：
            step 和 micro_step 值越小优先级越高；
            vpp 和 pp 在前向阶段值越小优先级越高，在非前向阶段值越大优先级越高；
            call_id 值越小优先级越高。
        """
        if not isinstance(other, GradAnomalyData):
            return NotImplemented

        self_train_stage = self.get_train_stage(self.tag_name)
        other_train_stage = self.get_train_stage(other.tag_name)

        def vpp_pp_comparator(anomaly):
            """
            Determine the priority rule for vpp and pp based on train stage
            Forward stage prefers smaller vpp and pp
            Other stages prefer larger vpp and pp
            """
            if self_train_stage == MonitorConst.FORWARD_STAGE:
                return anomaly.vpp_stage, anomaly.pp_stage
            else:
                return -anomaly.vpp_stage, -anomaly.pp_stage

        self_cmp = [self.step, self.micro_step, self_train_stage, *vpp_pp_comparator(self), self.call_id]
        other_cmp = [other.step, other.micro_step, other_train_stage, *vpp_pp_comparator(other), other.call_id]
        return self_cmp < other_cmp

    def __le__(self, other):
        if not isinstance(other, GradAnomalyData):
            return NotImplemented
        return self == other or self < other

    @staticmethod
    def get_train_stage(tag_name):
        """
        :param tag_name: "0:fc2.input:0/rank0/actv", "0:fc1.weight/rank0/post_grad", "0:fc2.weight/rank0/exp_avg_sq"
        :return: int, if forward return 0; if backward return 1; if optimizer return 2
        """
        key_ = tag_name.split("/")[-1]
        return MonitorConst.TRAIN_STAGE.get(key_, MonitorConst.DEFAULT_STAGE)

    def to_dict(self):
        return self.__dict__

    def get_key(self):
        # 0:1.self_attention.core_attention_flash_0/rank0/input_grad
        return ''.join([str(self.tag_name), "_step_", str(self.step), "_call_", str(self.call_id)])


class AnomalyDataWriter:
    """
    异常数据写入类，负责将异常数据写入到JSON文件中。
    """

    def __init__(self, dump_path, rank) -> None:
        self.dump_path = dump_path
        self.dump_rank_dir = os.path.join(self.dump_path, f"rank{rank}")
        self.json_path = os.path.join(self.dump_rank_dir, MonitorConst.ANOMALY_JSON)

    @staticmethod
    def get_anomaly_dict(anomalies):
        """将GradAnomalyData列表转换为json"""
        anomalies_json = {}
        for anomaly in anomalies:
            anomalies_json.update({anomaly.get_key(): anomaly.to_dict()})
        return anomalies_json

    def init_detected_json(self):
        """初始化落盘文件"""
        create_directory(self.dump_rank_dir)

        if os.path.exists(self.json_path):
            check_file_or_directory_path(self.json_path, isdir=False)
            logger.warning(f"The existing file will be deleted: {self.json_path}.")
            remove_path(self.json_path)
        save_json(self.json_path, {}, indent=1)

    def write_detected_json(self, anomalies):
        """
        落盘异常数据
        Args:
        anomalies: GradAnomalyData对象列表
        """
        anomalies_json = self.get_anomaly_dict(anomalies)
        if anomalies_json:
            logger.info(f"{MonitorConst.ANOMALY_JSON} is at {self.dump_rank_dir}.")

            data_to_write = load_json(self.json_path) if os.path.exists(self.json_path) else {}
            data_to_write.update(anomalies_json)
            save_json(self.json_path, data_to_write, indent=1)


class AnomalyDataLoader:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    @staticmethod
    def create_instances_from_dict(anomalies_dict: dict):
        instances = []
        for values in anomalies_dict.values():
            try:
                instances.append(GradAnomalyData(**values))
            except KeyError as e:
                logger.warning(f"Missing key in anomaly data: {e}.")
            except Exception as e:
                logger.warning(f"Value error when creating a GradAnomalyData instance: {e}.")
        return instances

    def get_anomalies_from_jsons(self):
        """遍历文件夹,从rankK/anomaly.json中读取异常数据
        return: anomalies: GradAnomalyData对象列表
        """
        anomalies = []
        check_file_or_directory_path(self.data_path, isdir=True)
        for rank_dir in os.listdir(self.data_path):
            rank_path = os.path.join(self.data_path, rank_dir)
            if not os.path.isdir(rank_path):
                continue
            json_path = os.path.join(rank_path, MonitorConst.ANOMALY_JSON)
            if not os.path.exists(json_path):
                continue
            data_anomalies = load_json(json_path)
            instances = self.create_instances_from_dict(data_anomalies)
            anomalies.extend(instances)
        return anomalies


class AnomalyAnalyse:
    def __init__(self) -> None:
        self.sorted_anomalies = []

    def get_range_top_k(self, topk, step_list, anomalies):
        """
        获取前topk个step_list范围内的异常。
        """
        if not step_list:
            filtered_anomalies = anomalies
        else:
            filtered_anomalies = [anomaly for anomaly in anomalies if anomaly.step in step_list]
        if topk >= len(filtered_anomalies):
            self.sorted_anomalies = sorted(filtered_anomalies)
        else:
            self.sorted_anomalies = list(heapq.nsmallest(topk, filtered_anomalies))
        return self.sorted_anomalies

    def rewrite_sorted_anomalies(self, output_path):
        """
        将排序后的异常数据重新落盘
        """
        check_file_or_directory_path(output_path, isdir=True)

        sorted_data = AnomalyDataWriter.get_anomaly_dict(self.sorted_anomalies)
        logger.info(f"{MonitorConst.ANALYSE_JSON} is at {output_path}.")
        json_path = os.path.join(output_path, MonitorConst.ANALYSE_JSON)
        if os.path.exists(json_path):
            logger.warning(f"The existing file will be deleted: {json_path}.")
            remove_path(json_path)
        save_json(json_path, sorted_data, indent=1)


def _get_step_and_stop(args):
    try:
        step_list = ast.literal_eval(args.step_list)
        if not isinstance(step_list, list):
            raise ValueError(f"{args.step_list} is not a list.")
    except (ValueError, SyntaxError, RecursionError) as e:
        raise ValueError("The step list must be a resolvable list type.") from e
    if args.top_k_number <= 0:
        raise ValueError("The top k number must be greater than 0.")
    return step_list, args.top_k_number


def _anomaly_analyse():
    args = _get_parse_args()
    step_list, top_k_number = _get_step_and_stop(args)
    loader = AnomalyDataLoader(args.data_path_dir)
    anomalies = loader.get_anomalies_from_jsons()
    analyser = AnomalyAnalyse()
    top_anomalies = analyser.get_range_top_k(top_k_number, step_list, anomalies)
    analyser.rewrite_sorted_anomalies(args.out_path if args.out_path else args.data_path_dir)

    logger.info(f"Top {top_k_number} anomalies are listed as follows:")
    for index, anomaly in enumerate(top_anomalies):
        logger.info(f"{index}: {anomaly.message}")


def _get_parse_args():
    parser = MindStudioArgumentParser(prog="anomaly_processor")
    parser.add_argument(
        "-d",
        "--data_path",
        dest="data_path_dir",
        default="./",
        type=str,
        metavar="<DIR>",
        help="<Required> The anomaly detect result dictionary: generate from monitor tool.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_path",
        dest="out_path",
        default="",
        type=str,
        help="<optional> The analyse task result out path.",
        required=False,
    )
    parser.add_argument(
        "-k",
        "--topk",
        dest="top_k_number",
        default=8,
        type=int,
        help="<optional> Top K number of earliest anomalies.",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--step",
        dest="step_list",
        default="[]",
        type=str,
        help="<optional> Analyse which steps.",
        required=False,
    )
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    _anomaly_analyse()
    logger.info("Analyse task completed.")
