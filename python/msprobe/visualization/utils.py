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

import os
import re
import json
import pickle  # nosec B403
import time
from msprobe.core.common.file_utils import FileOpen, find_proc_dir
from msprobe.core.common.const import LazyConst, CompareConst, Const
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.compare.utils import check_and_return_dir_contents
from msprobe.core.common.utils import CompareException


def load_json_file(file_path):
    """
    加载json文件
    """
    try:
        with FileOpen(file_path, 'r') as f:
            file_dict = json.load(f)
            if not isinstance(file_dict, dict):
                return {}
            return file_dict
    except json.JSONDecodeError:
        return {}


def load_data_json_file(file_path):
    """
    加载dump.json中的data字段
    """
    return load_json_file(file_path).get(GraphConst.DATA_KEY, {})


def str2float(percentage_str):
    """
    百分比字符串转换转换为浮点型
    Args:
        percentage_str: '0.00%', '23.4%'
    Returns: float 0.00, 0.234
    """
    try:
        percentage_str = percentage_str.strip('%')
        return float(percentage_str) / 100
    except (ValueError, AttributeError):
        return 0


def get_step_or_rank_int(x: str, is_rank=False):
    """
    获取字符串rank{int}或者step{int}中的int值，如果x=rank或step，返回0
    """
    if x in [Const.RANK, Const.STEP] or x is None or x.startswith(Const.PROC):
        return 0
    description = Const.RANK if is_rank else Const.STEP
    try:
        x_int = int(x.replace(Const.RANK, "")) if is_rank else int(x.replace(Const.STEP, ""))
    except Exception as e:
        logger.error(f'The folder name format is incorrect, expected {description}+number, such as rank0, step1, etc.')
        raise RuntimeError from e
    return x_int


def check_directory_content(input_path):
    """
    检查input_path内容, 是否全是step{数字}命名的文件夹(例如step0), 或者全是rank{数字}命名的文件夹(例如rank0), 或者全是文件
    """
    contents = os.listdir(input_path)
    if not contents:
        raise ValueError(f'The path {input_path} is empty.')

    # 真实数据dump会有dump_tensor_data文件夹
    if os.path.exists(os.path.join(input_path, Const.DUMP_TENSOR_DATA)):
        return GraphConst.FILES

    # 检查是否全是文件
    if all(os.path.isfile(os.path.join(input_path, item)) for item in contents):
        return GraphConst.FILES

    # 单卡只有一个rank文件夹
    if contents == [Const.RANK] or contents[0].startswith(Const.PROC):
        return GraphConst.RANKS

    rank_pattern = re.compile(r'^rank\d+$')
    step_pattern = re.compile(r'^step\d+$')

    rank_all = True
    step_all = True

    for item in contents:
        item_path = os.path.join(input_path, item)
        if not os.path.isdir(item_path):
            continue
        if not rank_pattern.match(item):
            rank_all = False
        if not step_pattern.match(item):
            step_all = False

    if rank_all:
        return GraphConst.RANKS
    if step_all:
        return GraphConst.STEPS

    raise ValueError(
        "The input path content does not conform to the expected naming convention. "
        "It is expected to be all step{number} named folders (such as step0), "
        "all rank{number} named folders (such as rank0), or all files."
    )


def extract_rank_number(rank_str):
    try:
        return int(rank_str[4:])
    except ValueError:
        return 0


def sort_rank_number_strings(rank_number_strings):
    sorted_list = sorted(rank_number_strings, key=extract_rank_number)
    return sorted_list


def load_parallel_param(args):
    param_t = [args.rank_size, args.tp, args.pp]
    if any(not x for x in param_t):
        logger.error('In the graph merge task, parameters "tp/pp/rank_size" is required!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    if not args.golden_path and len(args.rank_size) != 1:
        logger.error('In the graph merge build task, the number of parameters "rank_size" to be filled in is either 1!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    if args.golden_path and len(args.rank_size) != 2:
        logger.error(
            'In the graph merge compare task, the number of parameters "rank_size" to be filled in is either 2!'
        )
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    if len(args.rank_size) == 1:
        if any(len(x) != 1 for x in param_t):
            logger.error(
                'In the graph merge build task, the number of parameters "tp/pp/rank_size" to be filled in is either 1!'
            )
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

        param_n = ParallelParam(
            args.rank_size[0],
            args.tp[0],
            args.pp[0],
            args.vpp[0] if args.vpp else 1,
            args.order[0] if args.order else 'tp-cp-ep-dp-pp',
        )
        return (param_n,)
    else:
        if any(len(x) != 2 for x in param_t):
            logger.error(
                'In the graph merge compare task, '
                'the number of parameters "tp/pp/rank_size" to be filled in is either 2!'
            )
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

        param_n = ParallelParam(
            args.rank_size[0],
            args.tp[0],
            args.pp[0],
            args.vpp[0] if len(args.vpp) == 2 else 1,
            args.order[0] if len(args.order) == 2 else 'tp-cp-ep-dp-pp',
        )
        param_b = ParallelParam(
            args.rank_size[1],
            args.tp[1],
            args.pp[1],
            args.vpp[1] if len(args.vpp) == 2 else 1,
            args.order[1] if len(args.order) == 2 else 'tp-cp-ep-dp-pp',
        )
        return param_n, param_b


def validate_parallel_param(parallel_param, dump_path, log_prefix='[NPU]'):
    pattern = re.compile(r'^[a-z\-]+$')
    params = [parallel_param.tp, parallel_param.pp, parallel_param.rank_size, parallel_param.vpp]
    ranks = check_and_return_dir_contents(dump_path, Const.RANK)
    if len(ranks) != parallel_param.rank_size:
        logger.error(
            f'{log_prefix} The parallel param "rank_size" error, '
            f'you set {parallel_param.rank_size} but expected {len(ranks)}.'
        )
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if any(x is None for x in params):
        logger.error(f'{log_prefix} The parallel params "tp/pp/rank_size" must not be null!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if any(isinstance(x, bool) for x in params):
        logger.error(f'{log_prefix} The parallel params "tp/pp/vpp/rank_size" must not be bool!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if any(not isinstance(x, int) for x in params):
        logger.error(f'{log_prefix} The parallel params "tp/pp/vpp/rank_size" must be int!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if any(x <= 0 for x in params):
        logger.error(f'{log_prefix} The parallel params "tp/pp/vpp/rank_size" must be greater than 0!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if parallel_param.tp > parallel_param.rank_size:
        logger.error(f'{log_prefix} The parallel param "tp" must be less than or equal to "rank_size"!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if parallel_param.pp > parallel_param.rank_size:
        logger.error(f'{log_prefix} The parallel param "pp" must be less than or equal to "rank_size"!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if parallel_param.rank_size % parallel_param.tp != 0:
        logger.error(f'{log_prefix} The parallel param "rank_size" must be divisible by "tp"!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if parallel_param.rank_size % parallel_param.pp != 0:
        logger.error(f'{log_prefix} The parallel param "rank_size" must be divisible by "pp"!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if parallel_param.tp * parallel_param.pp > parallel_param.rank_size:
        logger.error(f'{log_prefix} The parallel params "tp * pp" must be less than or equal to "rank_size"!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if parallel_param.vpp > 1 and parallel_param.pp < 2:
        logger.error(f'{log_prefix} When configuring the parallel param "vpp", the "pp" param must be greater than 1!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(parallel_param.order, str):
        logger.error(f'{log_prefix} The parallel params "order" must be of string type!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not pattern.match(parallel_param.order):
        logger.error(f'{log_prefix} The parallel params "order" must consist only of lowercase letters and "-"!')
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)


class ParallelParam:
    def __init__(self, rank_size, tp, pp, vpp=1, order='tp-cp-ep-dp-pp'):
        self.rank_size = rank_size
        self.tp = tp
        self.pp = pp
        self.vpp = vpp
        self.order = order


class ToolTip:
    MAX_DIFF = 'NPU与标杆API统计信息比对，最大值的差值'
    MIN_DIFF = 'NPU与标杆API统计信息比对，最小值的差值'
    MEAN_DIFF = 'NPU与标杆API统计信息比对，平均值的差值'
    NORM_DIFF = 'NPU与标杆API统计信息比对，2范数（平方根）的差值'
    MD5 = '数据MD5信息，用于比较两个数据信息是否完全一致'
    ONE_THOUSANDTH_ERR_RATIO = (
        'Tensor中的元素逐个与对应的标杆数据对比，相对误差小于千分之一的比例占总元素个数的比例，比例越接近1越好'
    )
    FIVE_THOUSANDTHS_ERR_RATIO = (
        'Tensor中的元素逐个与对应的标杆数据对比，相对误差小于千分之五的比例占总元素个数的比例，比例越接近1越好'
    )
    COSINE = (
        '通过计算两个向量的余弦值来判断其相似度，数值越接近于1说明计算出的两个张量越相似，实际可接受阈值为大于0.99。'
        '在计算中可能会存在nan，主要由于可能会出现其中一个向量为0'
    )
    MAX_ABS_ERR = '当最大绝对误差越接近0表示其计算的误差越小，实际可接受阈值为小于0.001'
    MAX_RELATIVE_ERR = (
        '当最大相对误差越接近0表示其计算的误差越小。'
        '当dump数据中存在0或Nan时，比对结果中最大相对误差则出现inf或Nan的情况，属于正常现象'
    )


class GraphConst:
    CONSTRUCT_FILE = 'construct.json'
    DUMP_FILE = 'dump.json'
    STACK_FILE = 'stack.json'
    ERROR_KEY = 'error_key'
    SUMMARY_COMPARE = 0
    MD5_COMPARE = 1
    REAL_DATA_COMPARE = 2
    STRUCTURE_COMPARE = 3
    JSON_NPU_KEY = 'NPU'
    JSON_BENCH_KEY = 'Bench'
    JSON_TIP_KEY = 'ToolTip'
    JSON_ROOT_KEY = 'root'
    JSON_NODE_KEY = 'node'
    JSON_DATA_KEY = 'dump_data_dir'
    JSON_TASK_KEY = 'task'
    DATA_KEY = 'data'
    ROUND_TH = 6
    JSON_INDEX_KEY = 'precision_index'
    MATCHED_DISTRIBUTED = 'matched_distributed'
    OVERFLOW_LEVEL = 'overflow_level'
    MAX_INDEX_KEY = 1
    MIN_INDEX_KEY = 0
    INPUT = '.input.'
    OUTPUT = '.output.'
    STR_MAX_LEN = 50
    MD5_INDEX_LIST = CompareConst.MD5_COMPARE_INDEX + [
        CompareConst.REQ_GRAD_CONSIST,
        CompareConst.RESULT,
        CompareConst.ERROR_MESSAGE,
    ]
    REAL_DATA_INDEX_LIST = LazyConst(
        lambda: CompareConst.ALL_COMPARE_INDEX
        + [CompareConst.REQ_GRAD_CONSIST, CompareConst.RESULT, CompareConst.ERROR_MESSAGE]
    )
    SUMMARY_INDEX_LIST = CompareConst.SUMMARY_COMPARE_INDEX + [
        CompareConst.REQ_GRAD_CONSIST,
        CompareConst.RESULT,
        CompareConst.ERROR_MESSAGE,
    ]
    APIS_BETWEEN_MODULES = 'Apis_Between_Modules'
    APIS_BETWEEN_MODULES_ALL_RANKS = 'Apis_Between_Modules_All_Ranks'
    NULL = 'null'
    NONE = 'None'
    VALUE = 'value'
    DESCRIPTION = 'description'
    ACCURACY_LEVEL = 'accuracy_level'
    COLORS = 'Colors'
    MICRO_STEPS = 'MicroSteps'
    OVERFLOW_CHECK = 'OverflowCheck'
    UNMATCHED = 'unmatched node'

    DUMP_MODE_TO_GRAPHCOMPARE_MODE_MAPPING = {
        Const.ALL: REAL_DATA_COMPARE,
        Const.SUMMARY: SUMMARY_COMPARE,
        Const.MD5: MD5_COMPARE,
        Const.STRUCTURE: STRUCTURE_COMPARE,
    }

    GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING = {
        REAL_DATA_COMPARE: Const.ALL,
        SUMMARY_COMPARE: Const.SUMMARY,
        MD5_COMPARE: Const.MD5,
        STRUCTURE_COMPARE: Const.STRUCTURE,
    }

    COMPARE_INDICATOR_TO_PRECISION_INDEX_MAPPING = {
        CompareConst.PASS: 0,
        CompareConst.WARNING: 0.5,
        CompareConst.ERROR: 1,
    }

    RANKS = 'ranks'
    STEPS = 'steps'
    FILES = 'files'

    SRC = 'src'
    DST = 'dst'

    BATCH_P2P = 'batch_isend_irecv'
    OP = 'op'
    PEER = 'peer'
    GROUP_ID = 'group_id'

    UNCERTAINTY_THRESHOLD = 1e-6
    REDUCE_OPERATIONS = ['reduce_scatter', 'all_reduce']

    IGNORE_PRECISION_INDEX = {
        'empty',
        'empty_like',
        'empty_with_format',
        'new_empty_strided',
        'new_empty',
        'empty_strided',
    }
    VPP_CHUNK_0 = '0'

    PBAR_TOTAL = 100
    BAR_FORMAT = "{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}] "
    PBAR_DESC_PREFIX = '[Graph] Overall Progress'
    BUILD_STAGES_TOTAL = 2
    COMPARE_STAGES_TOTAL = 5

    INTERSECTION = 'intersection'
    UNION = 'union'


def is_serializable(obj):
    """
    Check if an object is serializable
    """
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False
    except Exception as e:
        logger.error('Unexpected error occurred while pickling obj.')
        raise RuntimeError('Unexpected error occurred while pickling obj.') from e


class SerializableArgs:
    def __init__(self, args):
        for k, v in vars(args).items():
            if is_serializable(v):
                setattr(self, k, v)


def calculate_list(path_n, path_b, prefix_str=Const.RANK, mode=GraphConst.INTERSECTION):
    """
    计算列表，可取交集、并集
    """
    if prefix_str not in [Const.RANK, Const.STEP]:
        logger.error(
            f'Parameter "prefix_str" is expected to be {Const.RANK} or {Const.STEP}, but in reality it is {prefix_str}.'
        )
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if mode not in [GraphConst.INTERSECTION, GraphConst.UNION]:
        logger.error(
            f'Parameter "mode" is expected to be {GraphConst.INTERSECTION} or {GraphConst.UNION}, '
            f'but in reality it is {mode}.'
        )
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    npu_list = check_and_return_dir_contents(path_n, prefix_str, skip_wrong_dir=True)
    bench_list = check_and_return_dir_contents(path_b, prefix_str, skip_wrong_dir=True)

    if not npu_list:
        npu_proc_dir = find_proc_dir(path_n)
        if npu_proc_dir:
            return [os.path.basename(npu_proc_dir)]
        logger.error(f'Cannot get {prefix_str} in {path_n}.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if npu_list != bench_list:
        result_list = npu_list
        if mode == GraphConst.INTERSECTION:
            result_list = sort_rank_number_strings(list(set(npu_list) & set(bench_list)))
        elif mode == GraphConst.UNION:
            result_list = sort_rank_number_strings(list(set(npu_list) | set(bench_list)))

        if not result_list:
            logger.error(f'Two {prefix_str}s cannot obtain {mode}.')
            raise CompareException(CompareException.INVALID_PATH_ERROR)
        return result_list
    return npu_list


POST_DB_PROCESS = 0


class ProgressInfo:
    """
    前端监测此类属性获取进度信息
    """

    current_progress = 0
    process_running = True
    error_msg = []
    info_dict = {}
    print_progress_log = False

    @classmethod
    def reset(cls):
        cls.current_progress = 0
        cls.process_running = True
        cls.error_msg = []

    @classmethod
    def update_current_progress(cls, value):
        if cls.current_progress != value:
            cls.current_progress = value
            cls.print_info()

    @classmethod
    def update_process_running(cls, value):
        if cls.process_running != value:
            cls.process_running = value
            cls.print_info()

    @classmethod
    def update_error_msg(cls, value):
        cls.error_msg.append(value)
        cls.print_info()

    @classmethod
    def print_info(cls):
        if cls.print_progress_log:
            cls.info_dict["current_progress"] = cls.current_progress
            cls.info_dict["process_running"] = cls.process_running
            cls.info_dict["error_msg"] = cls.error_msg
            logger.info(f'PROGRESS: {json.dumps(cls.info_dict)}')


def get_log_msg_wrapper(fn):
    def decorated(self, msg):
        ProgressInfo.update_error_msg(msg)
        return fn(self, msg)

    return decorated


def monitor_progress(pbar_info, pbar, all_task_ids, is_parallel_merge=False):
    """监测所有任务进度（包括等待中的），取平均值更新总进度"""
    ProgressInfo.total_stage = pbar_info.stage_total
    task_progress = pbar_info.progress_dict
    for task_id in all_task_ids:
        task_progress[task_id] = 0
    # 图合并模式，总进度卡99%，剩余的1%给db后处理函数; 否则总进度卡98%，剩余的1%给db后处理函数，1%给分布式节点分析关联
    global POST_DB_PROCESS
    POST_DB_PROCESS = (
        pbar_info.total - pbar_info.step_total if is_parallel_merge else pbar_info.total - 2 * pbar_info.step_total
    )
    all_complete = False

    while not all_complete:
        if pbar_info.stop_monitor:
            break
        if pbar_info.wait_monitor:
            logger.info('Detected wait_monitor=True, the progress monitoring thread has entered a paused state...')
            while pbar_info.wait_monitor:
                time.sleep(0.1)
                if pbar_info.stop_monitor:
                    break
            if pbar_info.continue_monitor:
                logger.info('Detected continue_monitor=True, the progress monitoring thread resumes running.')
        # 总进度取所有进程进度的平均值，但不超过数据库后处理进度
        current_progress = min(sum(task_progress.values()) // len(task_progress), POST_DB_PROCESS)
        ProgressInfo.update_current_progress(current_progress // pbar_info.step_total)
        pbar.n = current_progress
        pbar.refresh()
        # 检查是否所有任务都已完成: 数据库后处理是否完成
        all_complete = POST_DB_PROCESS == pbar_info.total
        # 降低检查频率
        time.sleep(0.1)

    post_process_pbar(pbar_info, pbar)


def update_shared_dict(shared_dict, key, value, update_all=False):
    """
    进程函数，根据键添加或更新共享字典
    """
    if update_all:
        for k in shared_dict.keys():
            shared_dict[k] += value
    else:
        shared_dict[key] = shared_dict.get(key, 0) + value


def update_pbar_info(pbar_info, current_data, total_data, step_ratio=0.01, update_all=False):
    """
    带步长控制的进度条更新函数
    :param pbar_info: 进度条信息对象
    :param current_data: 当前进度位置
    :param total_data: 总数据量
    :param step_ratio: 更新步长比例（默认每1%更新一次）
    :param update_all
    """
    step = max(1, int(total_data * step_ratio))

    # 仅在最后一次/达到步长时执行更新
    if current_data != total_data and current_data % step != 0:
        return

    progress = (current_data * pbar_info.total // total_data) // pbar_info.stage_total
    progress += pbar_info.stage_progress * (pbar_info.current_stage_dict.get(pbar_info.task_id) - 1)
    if pbar_info.pbar is not None:
        if progress == pbar_info.total:
            # 卡99%，剩余的1%给db后处理函数
            progress -= 1
        pbar_info.pbar.n = progress
        pbar_info.pbar.refresh()
        ProgressInfo.update_current_progress(progress)
    elif pbar_info.progress_dict is not None:
        if update_all:
            for task_id in pbar_info.progress_dict.keys():
                pbar_info.progress_dict[task_id] = progress
        else:
            pbar_info.progress_dict[pbar_info.task_id] = progress


def post_process_db_pbar(pbar_info, is_parallel_merge=False):
    if pbar_info.progress_dict is not None:
        global POST_DB_PROCESS
        POST_DB_PROCESS = pbar_info.total if is_parallel_merge else pbar_info.total - 1
    elif pbar_info.pbar is not None:
        pbar_info.pbar.n = pbar_info.total
        pbar_info.pbar.refresh()
        ProgressInfo.update_current_progress(pbar_info.total)


def post_process_pbar(pbar_info, pbar):
    pbar.n = pbar_info.total
    ProgressInfo.update_current_progress(pbar_info.total // pbar_info.step_total)
    pbar.set_description("[Graph] All tasks completed")
    pbar.refresh()


def update_progress_info(proc, progress_info: ProgressInfo):
    """
    通过管道获取日志，解析日志进度信息，更新进度类，供前端调用
    """
    try:
        while True:
            line = proc.stdout.readline()
            if line == '':
                if proc.poll() is not None:
                    break
            else:
                line = line.strip()
                if 'PROGRESS: ' in line:
                    line = line.split('PROGRESS: ')[-1]
                    info_dict = json.loads(line)
                    progress_info.current_progress = info_dict.get('current_progress')
                    progress_info.process_running = info_dict.get('process_running')
                    progress_info.error_msg = info_dict.get('error_msg')
            time.sleep(0.1)
    finally:
        proc.stdout.close()
