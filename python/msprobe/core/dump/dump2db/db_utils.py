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

from collections import defaultdict

from msprobe.core.common.const import Data2DBConst, Const
from msprobe.core.common.db_manager import DBManager, TrendSql
from msprobe.core.common.log import logger


def parse_full_key(full_key):
    """
    Module.blocks.3.layer.1.conv2.Conv2d.input.2
    """
    tags = set()
    parts = full_key.split('.')
    if len(parts) < 3:
        logger.debug(f"parse_full_key: key '{full_key}' has less than 3 parts,returning empty set of tags")
        return set()
    # 关联index标签 full_key中最后一部分，如 'input.2'
    tags.add((".".join(parts[-2:]), Data2DBConst.TAG_INDEX))

    # parameters类型打默认标签，如 Module.layer1.parameters.weight
    # 用 in 判断而非 parts[-2]，兼容 param_name 含点/数字的场景（如 parameters.mlp.0.weight）
    if Const.PARAMS in parts:
        tags.add((Const.PARAMS, Data2DBConst.TAG_DEFAULT))

    # 关联默认标签
    filtered_parts = []
    is_module = False
    for part in parts[:-2]:
        if part not in Data2DBConst.DEFAULT_TAGS:
            filtered_parts.append(part)
        else:
            tags.add((part, Data2DBConst.TAG_DEFAULT))
            is_module = True

    if not filtered_parts:
        return set()
    if not is_module:
        api_tag = (".".join(filtered_parts), Data2DBConst.TAG_FUNCTION)
        tags.add(api_tag)
        return tags
    # module级别拆分layer标签
    processed_indices = [-1]
    for i, part in enumerate(filtered_parts):
        if part.isdigit():
            if i == 0 or i - 1 in processed_indices:  # 确保数字前面有非数字部分, 组成layer标签
                processed_indices.append(i)
                continue
            layer_tag = (f"{filtered_parts[i - 1]}.{filtered_parts[i]}", Data2DBConst.TAG_LAYER)
            tags.add(layer_tag)

            before_layer = ".".join(filtered_parts[processed_indices[-1] + 1 : i - 1])
            if before_layer:
                module_tag = (before_layer, Data2DBConst.TAG_MODULE)
                tags.add(module_tag)
            processed_indices.append(i)
    if processed_indices[-1] != len(filtered_parts) - 1 and ".".join(filtered_parts[processed_indices[-1] + 1 :]):
        module_tag = (".".join(filtered_parts[processed_indices[-1] + 1 :]), Data2DBConst.TAG_MODULE)
        tags.add(module_tag)
    return tags


class DumpDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_manager = DBManager(db_path)

        # 缓存
        self._metric_id_cache = {}  # metric_name -> metric_id
        self._stats_cache = {}  # metric_name -> stats list
        self._processed_targets = defaultdict(dict)  # 记录所有metric_id下的targets
        self._new_targets = defaultdict(list)  # 记录新增targets
        self._init_schema()

    def init_global_stats_data(self, config):
        """初始化全局统计表数据"""
        # 准备插入数据
        self.db_manager.execute_sql(TrendSql.create_global_stats_table(config))

        column_values = []
        for _, column_value in config.items():
            if isinstance(column_value, (list, set)):
                column_values.append(",".join(column_value))
            elif isinstance(column_value, (int, float)):
                column_values.append(column_value)
        self.db_manager.insert_data(
            "global_stats",
            [
                column_values,
            ],
        )

    def get_metric_id(self, metric_name: str):
        """Get metric ID by name"""
        if metric_name in self._metric_id_cache:
            return self._metric_id_cache[metric_name]
        result = self.db_manager.select_data(
            table_name="monitoring_metrics", columns=["metric_id"], where={"metric_name": metric_name}
        )
        metric_id = result[0]["metric_id"] if result else None
        self._metric_id_cache[metric_name] = metric_id
        return metric_id

    def cache_targets(self, target, metric_id):
        if target not in self._processed_targets[metric_id]:
            self._processed_targets[metric_id][target] = {'id': 0}
            self._new_targets[metric_id].append(target)
        return self._processed_targets[metric_id][target]

    def batch_insert_targets(self):
        """批量插入targets并通过rowid范围获取ID"""

        for metric_id in sorted(self._new_targets.keys()):
            targets = self._new_targets[metric_id]
            if not targets:
                continue
            self.db_manager.insert_data(
                "monitoring_targets", list(targets), key_list=["target_name", "vpp_stage", "micro_step"]
            )
            results = self.db_manager.select_data(
                table_name="monitoring_targets", columns=["target_id", "target_name", "vpp_stage", "micro_step"]
            )
            # 更新缓存
            for row in results:
                cache_key = (row["target_name"], row["vpp_stage"], row["micro_step"])
                if cache_key in self._processed_targets[metric_id]:
                    self._processed_targets[metric_id][cache_key]['id'] = row["target_id"]
        self._new_targets = defaultdict(list)

    def batch_insert_data(self, batch_data):
        """批量插入数据"""
        if not batch_data:
            return
        # 刷新target id
        for row in batch_data:
            if len(row) < 3 or not isinstance(row[2], dict):
                return
            row[2] = row[2]["id"]

        self.db_manager.insert_data("trend_data", batch_data)

    def extract_tags_from_processed_targets(self):
        """从已处理的targets中提取标签并建立映射关系"""

        for metric_id in self._processed_targets:
            tag_info_to_id = {}
            mapping_data = []
            target_to_tags = {}
            new_tags_data = set()
            for target, info in self._processed_targets[metric_id].items():
                target_name, _, _ = target
                target_id = info.get("id")

                if target_id is None:
                    continue
                # 从target_name中提取标签
                tags = parse_full_key(target_name)
                new_tags_data = new_tags_data.union(tags)
                target_to_tags[target_id] = tags

            # 插入新标签
            if not new_tags_data:
                continue
            self.db_manager.insert_data(
                "monitoring_tags",
                [(tag_name, category, metric_id) for tag_name, category in new_tags_data],
                key_list=["tag_name", "category", "metric_id"],
            )

            # 重新获取tag_id用于映射
            tag_records = self.db_manager.select_data(
                "monitoring_tags", columns=["tag_id", "tag_name", "category", "metric_id"]
            )
            for row in tag_records:
                tag_info_to_id[(row["tag_name"], row["category"], row["metric_id"])] = row["tag_id"]

            for target_id, tags_list in target_to_tags.items():
                for tag_name, category in tags_list:
                    tag_id = tag_info_to_id.get((tag_name, category, metric_id))
                    if tag_id:
                        mapping_data.append((target_id, tag_id))

            # 插入映射关系
            if mapping_data:
                self.db_manager.insert_data("tag_target_mapping", mapping_data, key_list=["target_id", "tag_id"])

    def _init_schema(self) -> None:
        """Initialize database schema"""
        self.db_manager.execute_multi_sql(TrendSql.get_table_definition())
        self.db_manager.execute_sql(TrendSql.create_trend_table(Data2DBConst.ORDERED_STAT))
        for metric_name in Data2DBConst.METRICS:
            self.db_manager.insert_data("monitoring_metrics", [(metric_name,)], ["metric_name"])
