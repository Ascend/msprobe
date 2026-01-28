# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# ==============================================================================

import os
import json
import time
import threading
import subprocess
from abc import ABC, abstractmethod
from tensorboard.util import tb_logging
from ..utils.graph_utils import GraphUtils
from ..utils.global_state import GraphState
from ..utils.constant import Extension, DataType
from msprobe.core.common.file_utils import remove_path
from msprobe.visualization.utils import ProgressInfo,update_progress_info

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
    def load_converted_graph_data(logdir):
        # 没有   db文件，data为当logdir目录下的文件，进入构建模式
        def explore_dir(directory, depth=0, max_depth=3):
            """
            遍历指定目录下的文件和文件夹，直到给定的最大深度。
            
            :param directory: 要遍历的目录路径。
            :param depth: 当前递归调用的深度，默认为0（初始调用）。
            :param max_depth: 深度上限，即需要遍历的最深层级数。
            :return: 一个包含找到的目录名和yaml文件名及其相对路径的字典。
            """
            dirs = []
            yaml_files = []
            
            for content in os.listdir(directory):
                content_path = os.path.join(directory, content)
                relative_path = os.path.relpath(content_path, start=logdir)  # 获取相对于logdir的相对路径
                
                success, _ = GraphUtils.safe_check_load_file_path(content_path, os.path.isdir(content_path))
                
                if success and os.path.isdir(content_path):
                    dirs.append(relative_path)
                        
                if success and content.endswith('yaml'):
                    yaml_files.append(relative_path)
                    
                if depth < max_depth and os.path.isdir(content_path):
                    # 如果是目录且未达到最大深度，则递归进入该目录
                    sub_dirs, sub_yaml_files = explore_dir(content_path, depth + 1, max_depth)
                    dirs.extend(sub_dirs)
                    yaml_files.extend(sub_yaml_files)
            return dirs, yaml_files
        try:
            dirs, yaml_files = explore_dir(logdir)
            return {
            'success': True,
            'data':{
                'dirs':dirs,
                'yaml_files':yaml_files
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    @staticmethod
    def convert_to_graph(data):
        npu_path = data.get('npu_path','')
        bench_path = data.get('bench_path','')
        output_path = data.get('output_path','')
        layer_mapping = data.get('layer_mapping','')
        overflow_check = data.get('overflow_check',False)
        fuzzy_match = data.get('fuzzy_match',False)
        is_print_compare_log = data.get('is_print_compare_log',False)
        parallel_param_n = data.get('parallel_param_n', {})
        parallel_param_b = data.get('parallel_param_b', {})
        
        def call_path_api():
            ProgressInfo.reset()
            run_list=['msprobe','graph_visualize','-tp',npu_path,'-gp',bench_path,'-o',output_path,'-progress_log']
            if layer_mapping:
                run_list.append('-lm')
                run_list.append(layer_mapping)
            if overflow_check:
                run_list.append('-oc')
            if fuzzy_match:
                run_list.append('-fm')
            if is_print_compare_log:
                run_list.append('--is_print_compare_log')
            if parallel_param_n.get('rank_size', None):
                run_list.append('--rank_size')
                run_list.append(str(parallel_param_n.get('rank_size')))
            if parallel_param_b.get('rank_size', None):  
                run_list.append(str(parallel_param_b.get('rank_size')))
            if parallel_param_n.get('tp', None):
                run_list.append('--tp')
                run_list.append(str(parallel_param_n.get('tp')))
            if parallel_param_b.get('tp', None):
                run_list.append(str(parallel_param_b.get('tp')))
            if parallel_param_n.get('pp', None):
                run_list.append('--pp')
                run_list.append(str(parallel_param_n.get('pp')))
            if parallel_param_b.get('pp', None):
                run_list.append(str(parallel_param_b.get('pp')))
            if parallel_param_n.get('vpp', None):
                run_list.append('--vpp')
                run_list.append(str(parallel_param_n.get('vpp')))
            if parallel_param_b.get('vpp', None):
                run_list.append(str(parallel_param_b.get('vpp')))
            if parallel_param_n.get('order', None):
                run_list.append('--order')
                run_list.append(str(parallel_param_n.get('order')))
            if parallel_param_b.get('order', None):
                run_list.append(str(parallel_param_b.get('order')))
            print(run_list)
            proc=subprocess.Popen(run_list,stdout=subprocess.PIPE, text=True)
            update_progress_info(proc,ProgressInfo)
            
        thread= threading.Thread(target=call_path_api)
        thread.start()
        return {
            'success': True,
        }
        
    @staticmethod
    def get_convert_progress():
        try:
            while ProgressInfo.process_running:
                if len(ProgressInfo.error_msg) > 0:
                    process_info = {
                        'status': 'error',
                        'progress': ProgressInfo.current_progress,
                        'error': ProgressInfo.error_msg
                    }
                else:
                    process_info = {
                        'progress': ProgressInfo.current_progress,
                        'status': 'building',
                    }
                yield f"data: {json.dumps(process_info)}\n\n"
                time.sleep(0.1)

            # 正常完成
            process_info = {
                'progress': ProgressInfo.current_progress,
                'status': 'done',
            }
            yield f"data: {json.dumps(process_info)}\n\n"
            
        except Exception as e:
            error_info = {
                'status': 'error',
                'error': str(e),  # 或者更详细的：traceback.format_exc()
                'progress': getattr(ProgressInfo, 'current_progress', 0),
            }
            yield f"data: {json.dumps(error_info)}\n\n"

    
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
    
