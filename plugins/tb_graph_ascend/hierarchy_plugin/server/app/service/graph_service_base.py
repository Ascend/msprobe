# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
from abc import ABC, abstractmethod

from tensorboard.util import tb_logging
from ..utils.graph_utils import GraphUtils
from ..utils.global_state import GraphState
from ..utils.constant import Extension, DataType

logger = tb_logging.get_logger()
DB_EXT = Extension.DB.value
JSON_EXT = Extension.JSON.value
DB_TYPE = DataType.DB.value
JSON_TYPE = DataType.JSON.value


class GraphServiceStrategy(ABC):
    run = ''
    tag = ''

    def __init__(self, run, tag):
        self.run = run
        self.tag = tag

    @staticmethod
    def load_meta_dir():
        """
        Scan logdir for directories containing .vis(.db) files. If the directory contains .vis.db files,
        it is considered a db type and the .vis files are ignored. Otherwise, it is considered a json type.
        """
        logdir = GraphState.get_global_value('logdir')
        runs = GraphState.get_global_value('runs', {})
        first_run_tags = GraphState.get_global_value('first_run_tags', {})
        meta_dir = {}
        error_list = []
        success, error = GraphUtils.safe_check_load_file_path(logdir, True)
        if not success:
            error_list.append({
                'run': logdir,
                'tag': '',
                'info': f'Error logdir:  {str(error)}'
            })
            result = {
                'data': meta_dir,
                'error': error_list
            }
            return {'success': False, 'error': error_list}
        for root, _, files in GraphUtils.walk_with_max_depth(logdir, 2):
            run_abs = os.path.abspath(root)
            run = os.path.basename(run_abs)  # 不允许同名目录，否则有问题
            for file in files:
                if file.endswith(DB_EXT):
                    tag = file[:-len(DB_EXT)]
                    _, error = GraphUtils.safe_load_data(run_abs, file, True)
                    if error:
                        error_list.append({
                            'run': run,
                            'tag': tag,
                            'info': f'Error: {str(error)}'
                        })
                        logger.error(f'Error: File run:"{run_abs},tag:{tag}" is not accessible. Error: {error}')
                        continue
                    runs[run] = run_abs
                    meta_dir[run] = {'type': DB_TYPE, 'tags': [tag]}
                    break
                # 只取文件名，不包含扩展名
                if file.endswith(JSON_EXT):
                    tag = os.path.splitext(file)[0]
                    _, error = GraphUtils.safe_load_data(run_abs, file, True)
                    if error:
                        error_list.append({
                            'run': run,
                            'tag': tag,
                            'info': f'Error: {error}'
                        })
                        logger.error(f'Error: File run:"{run_abs},tag:{tag}" is not accessible. Error: {error}')
                        continue
                    runs[run] = run_abs
                    if meta_dir.get(run) is None:
                        meta_dir[run] = {'type': JSON_TYPE, 'tags': [tag]}
                    else:
                        meta_dir.get(run).get('tags').append(tag)
        meta_dir = GraphUtils.sort_data(meta_dir)
        for run, value in meta_dir.items():
            first_run_tags[run] = value.get('tags')[0] if value.get('tags') else ''
        GraphState.set_global_value('runs', runs)
        GraphState.set_global_value('first_run_tags', first_run_tags)
        result = {
            'data': meta_dir,
            'error': error_list
        }
        return result

    @abstractmethod
    def load_graph_data(self):
        pass

    @abstractmethod
    def load_graph_config_info(self):
        pass

    @abstractmethod
    def load_graph_all_node_list(self, meta_data):
        pass

    @abstractmethod
    def change_node_expand_state(self, node_info, meta_data):
        pass
        
    def search_node_by_precision(self, meta_data, values):
        pass
    
    def search_node_by_overflow(self, meta_data, values):
        pass
    
    @abstractmethod
    def get_node_info(self, node_info, meta_data):
        pass

    @abstractmethod
    def add_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_match_children):
        pass

    @abstractmethod
    def add_match_nodes_by_config(self, config_file_name, meta_data):
        pass

    @abstractmethod
    def delete_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_unmatch_children):
        pass
    
    @abstractmethod
    def update_precision_error(self, meta_data, filter_value):
        pass    

    @abstractmethod
    def update_colors(self, colors):
        pass

    @abstractmethod
    def save_matched_relations(self, meta_data):
        pass
    
