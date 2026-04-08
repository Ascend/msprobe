# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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
import stat

import numpy as np


class Const:
    """
    Class for const
    """
    TOOL_NAME = "msprobe"
    MD5_INDEX = "md5_index"
    MD5 = "md5"

    ipv4_pattern = "([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])(\.([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])){3}$"
    SEP = "."
    REGEX_PREFIX_MAX_LENGTH = 20
    REGEX_PREFIX_PATTERN = r"^[a-zA-Z0-9_-]+$"
    REGEX_FORWARD_BACKWARD = r'\.(forward|backward)\.'
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    STRING_BLACKLIST = r"^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]"
    COMMA = ","
    FLOAT_EPSILON = np.finfo(float).eps
    OFF = 'OFF'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    PROGRESS_TIMEOUT = 3000
    EXCEPTION_NONE = None
    JIT = 'Jit'
    PRIMITIVE_PREFIX = 'Primitive'
    DEFAULT_LIST = []
    DEFAULT_PATH = './'
    WHITE_LIST = 'white_list'
    BLACK_LIST = 'black_list'
    DUMP_TENSOR_DATA = 'dump_tensor_data'
    NONE = None
    THREE_SEGMENT = 3
    FOUR_SEGMENT = 4
    SIX_SEGMENT = 6
    SEVEN_SEGMENT = 7

    MAX_DEPTH = 400
    CPU_QUARTER = 4
    DUMP_MAX_DEPTH = 400

    EXTERN_INPUT_LIST_MAX_LEN = 100
    MAX_PROCESS_NUM = 128

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]
    AUTO = "auto"
    ONLINE_DUMP_MODE = [ALL, LIST, AUTO, OFF]
    SUMMARY = "summary"
    MD5 = "md5"
    HASH = "hash"
    XOR_CHECKSUM = "xor"
    VALUE = "value"
    SUMMARY_MODE = ["statistics", "md5", "xor"]

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
    OVERWRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    NUMPY_PATTERN = "*.npy"
    PT_SUFFIX = ".pt"
    PY_SUFFIX = ".py"
    INIT_PY = "init.py"
    ONE_GB = 1073741824  # 1 * 1024 * 1024 * 1024
    TEN_GB = 10737418240  # 10 * 1024 * 1024 * 1024
    ONE_MB = 1048576  # 1 * 1024 * 1024
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    DISTRIBUTED_PREFIX_LENGTH = 60
    # env dump path
    KWARGS = 'kwargs'
    INPUT = 'input'
    OUTPUT = 'output'
    INPUT_ARGS = 'input_args'
    INPUT_KWARGS = 'input_kwargs'
    GRAD_INPUT = 'grad_input'
    GRAD_OUTPUT = 'grad_output'
    PARAMS = 'parameters'
    PARAMS_GRAD = 'parameters_grad'
    DEBUG = 'debug'
    START = "start"
    STOP = "stop"
    ENV_ENABLE = "1"
    ENV_DISABLE = "0"
    MAX_SEED_VALUE = 4294967295  # 2**32 - 1
    STATISTICS = "statistics"
    TENSOR = "tensor"
    OVERFLOW_CHECK = "overflow_check"
    DIFF_CHECK = "diff_check"
    FREE_BENCHMARK = "free_benchmark"
    ACC_CHECK = "acc_check"
    STRUCTURE = "structure"
    EXCEPTION_DUMP = "exception_dump"
    DUMP_PRECISION_HIGH = "high"
    DUMP_PRECISION_LOW = "low"
    TASK_LIST = [TENSOR, STATISTICS, ACC_CHECK, STRUCTURE, OVERFLOW_CHECK, DIFF_CHECK, EXCEPTION_DUMP]
    TORCH_TASK_LIST = [TENSOR, STATISTICS, ACC_CHECK, STRUCTURE, DIFF_CHECK]
    DUMP_DATA_COLLECTION_LIST = [STATISTICS, TENSOR, STRUCTURE]
    DUMP_DATA_MODE_LIST = [ALL, INPUT, OUTPUT, FORWARD, BACKWARD]
    DUMP_PRECISION_LIST = [DUMP_PRECISION_LOW, DUMP_PRECISION_HIGH]
    LEVEL_L0 = "L0"
    LEVEL_L1 = "L1"
    LEVEL_L2 = "L2"
    LEVEL_MIX = "mix"
    LEVEL_DEBUG = "debug"
    LEVEL_LIST = [LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_MIX, LEVEL_DEBUG]
    RISK_LEVEL_CORE = "CORE"
    RISK_LEVEL_FOCUS = "FOCUS"
    RISK_LEVEL_LOW = "LOW"
    RISK_LEVEL_ALL = "ALL"
    RISK_LEVEL_LIST = [RISK_LEVEL_CORE, RISK_LEVEL_FOCUS, RISK_LEVEL_ALL]
    ATTR_NAME_PREFIX = "wrap_"
    ATTR_NAME_PREFIX_LEN = len(ATTR_NAME_PREFIX)
    KERNEL_DUMP = "kernel_dump"
    DATA = "data"
    PT_FRAMEWORK = "pytorch"
    MS_FRAMEWORK = "mindspore"
    MT_FRAMEWORK = "mindtorch"
    UNKNOWN_FRAMEWORK = "unknown"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
    BOOL_TYPE = [bool, np.uint8]
    INT_TYPE = [np.int32, np.int64]
    NPU = 'NPU'
    NPU_LOWERCASE = 'npu'
    CPU_LOWERCASE = 'cpu'
    GPU_LOWERCASE = 'gpu'
    CUDA_LOWERCASE = 'cuda'
    DEVICE = 'device'
    DISTRIBUTED = 'Distributed'
    DUMP_PREFIX = ["Distributed", "Functional", "Torch", "Tensor", "Mint", "MintFunctional", "Primitive",
                   "Aten", "VF", "NPU", "Jit"]
    MODULE_PREFIX = ["Module", "Cell"]
    FORWARD_NAME_SUFFIX = ".forward"

    DUMP_JSON_FILE = "dump_json_file"
    DEBUG_JSON_FILE = "debug_json_file"
    STACK_JSON_FILE = "stack_json_file"

    # struct json param
    ORIGIN_DATA = "origin_data"
    SCOPE = "scope"
    STACK = "stack"

    ATEN = "Aten"
    MODULE_WHITE_LIST = ["torch", "numpy"]

    FUNC_SKIP_LIST = ["construct", "__call__"]
    FILE_SKIP_LIST = ["msprobe", "MindSpeed"]
    DATA_TYPE_SKIP_LIST = ["Primitive", "Jit"]

    STACK_FILE_INDEX = 0
    STACK_FUNC_INDEX = 2
    STACK_FUNC_ELE_INDEX = 1

    SCOPE_ID_INDEX = -1
    SCOPE_DIRECTION_INDEX = -2
    TYPE_NAME_INDEX = -3
    PARAMS_GRAD_TYPE_NAME_INDEX = -2
    LAYER_NAME_INDEX = -4
    PARAMS_GRAD_NAME_INDEX = -3
    API_TYPE_INDEX = 0
    LEFT_MOVE_INDEX = -1
    RIGHT_MOVE_INDEX = 1
    LAST_INDEX = -1

    TOP_LAYER = "TopLayer"
    CELL = "Cell"
    MODULE = "Module"
    API = "api"
    CLASS = "class"
    PYNATIVE_MODE = "pynative"
    PYNATIVE_GRAPH_MODE = "pynative_graph"

    FRAME_FILE_LIST = ["site-packages/torch", "package/torch", "site-packages/mindspore", "package/mindspore"]
    INPLACE_LIST = [
        "broadcast", "all_reduce", "reduce", "all_gather", "gather", "scatter", "reduce_scatter",
        "_reduce_scatter_base", "_all_gather_base", "send", "recv", "irecv", "isend", "all_to_all_single", "all_to_all",
        "all_gather_into_tensor", "reduce_scatter_tensor"
    ]

    CONVERT = {
        "int32_to_int64": ["torch.int32", "torch.int64"],
        "int64_to_fp32": ["torch.int64", "torch.float32"]
    }

    CONVERT_API = {
        "int32_to_int64": ["cross_entropy"],
        "int64_to_fp32": ["histc"]
    }

    FA_SPECIAL_SPARSE_MODE = [2, 3, 4]

    FILL_CHAR_NUMS = 50
    TOOL_ENDS_SUCCESSFULLY = f"{TOOL_NAME} ends successfully."

    WITHOUT_CALL_STACK = "The call stack retrieval failed."
    STACK_FILTER_KEYWORDS = ["msprobe/core", "msprobe/pytorch", "msprobe/mindspore"]
    CALL_STACK_FLAG = "api_dump/api_registry"
    NEW_STACK_FLAG = "0"

    STEP = "step"
    RANK = "rank"
    PROC = "proc"
    HYPHEN = "-"
    STEP_RANK_MINIMUM_VALUE = 0
    STEP_RANK_MAXIMUM_VALUE = int(1e6)

    # data type const
    TORCH_INT_DTYPE = ["torch.int8", "torch.int32", "torch.int64"]
    TORCH_FLOAT_DTYPE = ["torch.bfloat16", "torch.float16", "torch.float32", "torch.float64"]
    FLOAT16 = "Float16"
    FLOAT32 = "Float32"
    BFLOAT16 = "BFloat16"
    TORCH_FLOAT16 = "torch.float16"
    TORCH_FLOAT32 = "torch.float32"
    TORCH_BFLOAT16 = "torch.bfloat16"

    TYPE = 'type'
    DTYPE = 'dtype'
    SHAPE = 'shape'
    LAYOUT = 'layout'
    SHARD_SIZE = 'hsdp_shard_size'
    STACK_INFO = 'stack_info'
    MAX = 'Max'
    MIN = 'Min'
    MEAN = 'Mean'
    NORM = 'Norm'
    DATA_NAME = 'data_name'
    STATE = 'state'
    REQ_GRAD = 'requires_grad'
    API_ORIGIN_NAME = 'api_origin_name'
    TENSOR_STAT_INDEX = 'tensor_stat_index'
    SUMMARY_METRICS_LIST = [MAX, MIN, MEAN, NORM]

    CODE_STACK = 'Code Stack'
    OP_NAME = 'Op Name'
    SCOPE_NAME = 'Scope Name'
    CODE_STACKS = 'Code Stacks'
    FILE_PATH = 'File Path'
    NEW_LINE = '\n'
    CSV_NEWLINE_SEPARATOR = ',\n'
    # 分隔符常量
    SCOPE_SEPARATOR = "/"
    REPLACEMENT_CHARACTER = "_"
    PIPE_SEPARATOR = "|"

    FORWARD_PATTERN = SEP + FORWARD + SEP
    BACKWARD_PATTERN = SEP + BACKWARD + SEP

    OPTIMIZER = "optimizer"
    CLIP_GRAD = "clip_grad"
    END_PREFIX = "end_"

    TENSOR_STAT_LEN = 2

    TENSOR_TYPE = "torch.Tensor"
    DTENSOR_TYPE = "torch.distributed.tensor.DTensor"
    FAKE_TENSOR_TYPE = "torch._subclasses.fake_tensor.FakeTensor"
    AC_TENSOR_TYPE = "torch.distributed._functional_collectives.AsyncCollectiveTensor"
    NESTED_TENSOR_TYPE = "torch.nested._internal.nested_tensor.NestedTensor"
    BATCHED_TENSOR_TYPE = "BatchedTensor"
    GRADTRACKING_TENSOR_TYPE = "GradTrackingTensor"

    SUPPORT_API_FILE_NAME = "support_wrap_ops.yaml"
    BUILTIN_IGNORE_API_FILE_NAME = "builtin_ignore_ops.yaml"

    API_ATTR_LIST = ["__name__", "default"]

    PT_API_TYPE_FUNCTIONAL = "functional"
    PT_API_TYPE_TENSOR = "tensor"
    PT_API_TYPE_TORCH = "torch"
    PT_API_TYPE_VF = "_VF"
    PT_API_TYPE_NPU = "torch_npu"
    PT_API_TYPE_ATEN = "aten"
    PT_API_TYPE_DIST = "distributed"
    PT_API_TYPE_DIST_C10D = "distributed_c10d"
    PT_API_TYPE_NPU_DIST = "npu_distributed"
    PT_API_TYPE_MINDSPEED = "mindspeed"

    MS_API_TYPE_OPS = "ops"
    MS_API_TYPE_TENSOR = "tensor"
    MS_API_TYPE_STUB_TENSOR = "stubtensor"
    MS_API_TYPE_MINT = "mint.ops"
    MS_API_TYPE_MINT_FUNC = "mint.nn.functional"
    MS_API_TYPE_COM = "communication.comm_func"
    MS_API_TYPE_MINT_DIST = "mint.distributed"

    FUNCTIONAL_API_TYPE_PREFIX = "Functional"
    TENSOR_API_TYPE_PREFIX = "Tensor"
    DIST_API_TYPE_PREFIX = "Distributed"

    TORCH_API_TYPE_PREFIX = "Torch"
    NPU_API_TYPE_PREFIX = "NPU"
    ATEN_API_TYPE_PREFIX = "Aten"
    VF_API_TYPE_PREFIX = "VF"
    MINDSPEED_API_TYPE_PREFIX = "MindSpeed"

    MINT_API_TYPE_PREFIX = "Mint"
    MINT_FUNC_API_TYPE_PREFIX = "MintFunctional"
    MINT_DIST_API_TYPE_PREFIX = "MintDistributed"

    SUPPORT_API_DICT_KEY_MAP = {
        PT_FRAMEWORK: {
            PT_API_TYPE_FUNCTIONAL: PT_API_TYPE_FUNCTIONAL,
            PT_API_TYPE_TENSOR: PT_API_TYPE_TENSOR,
            PT_API_TYPE_TORCH: PT_API_TYPE_TORCH,
            PT_API_TYPE_VF: PT_API_TYPE_VF,
            PT_API_TYPE_NPU: PT_API_TYPE_NPU,
            PT_API_TYPE_ATEN: PT_API_TYPE_ATEN,
            PT_API_TYPE_DIST: PT_API_TYPE_DIST,
            PT_API_TYPE_DIST_C10D: PT_API_TYPE_DIST,
            PT_API_TYPE_NPU_DIST: PT_API_TYPE_NPU_DIST,
            PT_API_TYPE_MINDSPEED: PT_API_TYPE_MINDSPEED
        },
        MS_FRAMEWORK: {
            MS_API_TYPE_OPS: MS_API_TYPE_OPS,
            MS_API_TYPE_TENSOR: MS_API_TYPE_TENSOR,
            MS_API_TYPE_STUB_TENSOR: MS_API_TYPE_TENSOR,
            MS_API_TYPE_MINT: MS_API_TYPE_MINT,
            MS_API_TYPE_MINT_FUNC: MS_API_TYPE_MINT_FUNC,
            MS_API_TYPE_COM: MS_API_TYPE_COM,
            MS_API_TYPE_MINT_DIST: MS_API_TYPE_MINT_DIST
        },
        MT_FRAMEWORK: {
            PT_API_TYPE_FUNCTIONAL: PT_API_TYPE_FUNCTIONAL,
            PT_API_TYPE_TENSOR: PT_API_TYPE_TENSOR,
            PT_API_TYPE_TORCH: PT_API_TYPE_TORCH,
            PT_API_TYPE_NPU: PT_API_TYPE_NPU,
            PT_API_TYPE_DIST: PT_API_TYPE_DIST
        }
    }

    API_DATA_PREFIX = {
        PT_FRAMEWORK: {
            PT_API_TYPE_FUNCTIONAL: FUNCTIONAL_API_TYPE_PREFIX,
            PT_API_TYPE_TENSOR: TENSOR_API_TYPE_PREFIX,
            PT_API_TYPE_TORCH: TORCH_API_TYPE_PREFIX,
            PT_API_TYPE_VF: VF_API_TYPE_PREFIX,
            PT_API_TYPE_NPU: NPU_API_TYPE_PREFIX,
            PT_API_TYPE_ATEN: ATEN_API_TYPE_PREFIX,
            PT_API_TYPE_DIST: DIST_API_TYPE_PREFIX,
            PT_API_TYPE_DIST_C10D: DIST_API_TYPE_PREFIX,
            PT_API_TYPE_NPU_DIST: DIST_API_TYPE_PREFIX,
            PT_API_TYPE_MINDSPEED: MINDSPEED_API_TYPE_PREFIX
        },
        MS_FRAMEWORK: {
            MS_API_TYPE_OPS: FUNCTIONAL_API_TYPE_PREFIX,
            MS_API_TYPE_TENSOR: TENSOR_API_TYPE_PREFIX,
            MS_API_TYPE_STUB_TENSOR: TENSOR_API_TYPE_PREFIX,
            MS_API_TYPE_MINT: MINT_API_TYPE_PREFIX,
            MS_API_TYPE_MINT_FUNC: MINT_FUNC_API_TYPE_PREFIX,
            MS_API_TYPE_COM: DIST_API_TYPE_PREFIX,
            MS_API_TYPE_MINT_DIST: MINT_DIST_API_TYPE_PREFIX
        },
        MT_FRAMEWORK: {
            PT_API_TYPE_FUNCTIONAL: FUNCTIONAL_API_TYPE_PREFIX,
            PT_API_TYPE_TENSOR: TENSOR_API_TYPE_PREFIX,
            PT_API_TYPE_TORCH: TORCH_API_TYPE_PREFIX,
            PT_API_TYPE_NPU: NPU_API_TYPE_PREFIX,
            PT_API_TYPE_DIST: DIST_API_TYPE_PREFIX
        }
    }

    def _fused_adamw_(
            self,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            *,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
            maximize,
            grad_scale=None,
            found_inf=None
    ):
        pass

    API_WITH_SELF_ARG = {
        'Torch._fused_adamw_': _fused_adamw_
    }

    ASCEND = "ASCEND"
    MATCH_MODE_NAME = "pure name"
    MATCH_MODE_MAPPING = "mapping"
    MATCH_MODE_SIMILARITY = "similarity"
    CONFIG_CHECK_PASS = "pass"
    CONFIG_CHECK_WARNING = "warning"
    CONFIG_CHECK_ERROR = "error"

    MIX_DUMP_NAMES = {'graph', 'pynative'}

    MEGATRON_MICRO_STEP_NUMBER = 'megatron_micro_step_number'

    TORCHAIR_FX_DATA_DIRECTORY = 'msprobe_fx_dump'
    TORCHAIR_GE_DATA_DIRECTORY = 'msprobe_ge_dump'

    FULL_OP_NAME = "full_op_name"
    DIRECTION = "direction"
    CALL_DIRECTION = "call_direction"
    OP_NO_NUMBER = "op_no_number"
    FORWARD_CALL_ORDER = "forward_call_order"
    BACKWARD_CALL_ORDER = "backward_call_order"
    SUFFIX = "suffix"

    LAYER = "layer"
    MODULE_CLASS = "module_class"
    MODULE_LEN = "module_len"
    MODULE_PART_PREFIX = "module_part_prefix"
    MODULE_LAST = "module_last"
    MODULE_MAPPING = "module_mapping"
    MODULE_LAST_MAPPING = "module_last_mapping"

    FSDP = "fsdp"
    MEGATRON = "megatron"


class Data2DBConst:
    BATCH_SIZE = 5000
    FORWARD = "forward"
    BACKWARD = "backward"
    RECOMPUTE = "recompute"
    PARAMETERS_GRAD = "parameters_grad"
    METRICS = [FORWARD, BACKWARD, RECOMPUTE, PARAMETERS_GRAD]
    DEFAULT_TAGS = ["Module", "Cell"]
    ROOT_NODE_PREFIX = ["Module", "Cell", "optimizer"]
    ORDERED_STAT = ["norm", "max", "min", "mean"]
    TAG_DEFAULT = "default"
    TAG_INDEX = "index"
    TAG_LAYER = "layer"
    TAG_MODULE = "module"
    TAG_FUNCTION = "function"
    SUPPORT_TYPE = ["torch.Tensor", "mindspore.Tensor", "mindtorch.Tensor"]
    SUPPORT_DTYPE = [Const.FLOAT16,
                     Const.FLOAT32,
                     Const.BFLOAT16,
                     Const.TORCH_FLOAT16,
                     Const.TORCH_FLOAT32,
                     Const.TORCH_BFLOAT16]
    DB_DUMP = "dump_data.trend.db"
    DB_MONITOR = "monitor_data.trend.db"
    # csv2db
    DEFAULT_INT_VALUE = 0
    MAX_PROCESS_NUM = 128
    OP_TRENDVIS_SUPPORTED = [
        "norm", "min", "max", "zeros", "nans", "mean",
        "entropy", "softmax_max", "sr", "kernel_norm", "std_x", "jacobian",
        "proxy", "token_similarity"
    ]
    METRICS_TRENDVIS_SUPPORTED = [
        "actv", "actv_grad", "exp_avg", "exp_avg_sq",
        "grad_unreduced", "grad_reduced", "param_origin", "param_updated", "other"
    ]
    FILE_BATCH_SIZE = 100
    MAX_FLOAT_VALUE = 1e9
    MIN_FLOAT_VALUE = -1e9


class CompareConst:
    """
    Class for compare module const
    """
    SPACE = " "
    NAME = "Name"
    # compare result column name
    NPU_NAME = "NPU Name"
    BENCH_NAME = "Bench Name"
    NPU_DTYPE = "NPU Dtype"
    BENCH_DTYPE = "Bench Dtype"
    NPU_SHAPE = "NPU Tensor Shape"
    BENCH_SHAPE = "Bench Tensor Shape"
    NPU_CSV_FILE = "NPU CSV File"
    BENCH_CSV_FILE = "Bench CSV File"
    NPU_MAX = "NPU max"
    NPU_MIN = "NPU min"
    NPU_MEAN = "NPU mean"
    NPU_NORM = "NPU l2norm"
    NPU_SUMMARY = [NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM]
    NPU_P2POP_PEER = "NPU P2POp peer"

    BENCH_MAX = "Bench max"
    BENCH_MIN = "Bench min"
    BENCH_MEAN = "Bench mean"
    BENCH_NORM = "Bench l2norm"
    BENCH_SUMMARY = [BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM]
    MAX_DIFF = "Max diff"
    MIN_DIFF = "Min diff"
    MEAN_DIFF = "Mean diff"
    NORM_DIFF = "L2norm diff"
    COSINE = "Cosine"
    EUC_DIST = "EucDist"
    MAX_ABS_ERR = "MaxAbsErr"
    MAX_RELATIVE_ERR = "MaxRelativeErr"
    MIN_RELATIVE_ERR = "MinRelativeErr"
    MEAN_RELATIVE_ERR = "MeanRelativeErr"
    NORM_RELATIVE_ERR = "NormRelativeErr"
    REQ_GRAD_CONSIST = "Requires_grad Consistent"
    NPU_REQ_GRAD = "NPU Requires_grad"
    BENCH_REQ_GRAD = "Bench Requires_grad"
    ACCURACY = "Accuracy Reached or Not"
    STACK = "NPU_Stack_Info"
    DATA_NAME = "Data_name"
    ERROR_MESSAGE = "Err_message"
    ONE_THOUSANDTH_ERR_RATIO = "One Thousandth Err Ratio"
    FIVE_THOUSANDTHS_ERR_RATIO = "Five Thousandths Err Ratio"
    NPU_MD5 = "NPU MD5"
    BENCH_MD5 = "BENCH MD5"
    RESULT = "Result"
    MAGNITUDE = 0.5
    OP_NAME = "op_name"
    OP_NAME_UPDATE = "op_name_update"
    STRUCT = "struct"
    INPUT_STRUCT = "input_struct"
    KWARGS_STRUCT = "kwargs_struct"
    OUTPUT_STRUCT = "output_struct"
    PARAMS_STRUCT = "params_struct"
    PARAMS_GRAD_STRUCT = "params_grad_struct"
    DEBUG_STRUCT = "debug_struct"
    SUMMARY = "summary"
    COMPARE_RESULT = "compare_result"
    COMPARE_MESSAGE = "compare_message"
    MAX_EXCEL_LENGTH = 1048500
    YES = "Yes"
    NO = "No"
    STATISTICS_INDICATOR_NUM = 4
    EPSILON = 1e-10
    COMPARE_ENDS_SUCCESSFULLY = "msprobe compare ends successfully."
    DEFAULT_RATIO_VALUE = 10000
    THOUSANDTH_PASS_VALUE = 0.999
    ZERO_SHAPE = '(0,)'

    BENCHMARK_COMPARE_ALGORITHM_NAME = "标杆比对法"
    ULP_COMPARE_ALGORITHM_NAME = "ULP误差比对法"
    BINARY_CONSISTENCY_ALGORITHM_NAME = "二进制一致法"
    ABSOLUTE_THRESHOLD_ALGORITHM_NAME = "绝对阈值法"
    THOUSANDTH_STANDARD_ALGORITHM_NAME = "双千指标法"
    ACCUMULATIVE_ERROR_COMPARE_ALGORITHM_NAME = "累积误差比对法"

    ABSOLUTE_THRESHOLD = 'absolute_threshold'
    BINARY_CONSISTENCY = 'binary_consistency'
    ULP_COMPARE = 'ulp_compare'
    THOUSANDTH_STANDARD = 'thousandth_threshold'
    BENCHMARK = 'benchmark'
    ACCUMULATIVE_ERROR_COMPARE = 'accumulative_error_compare'

    SMALL_VALUE_ERR_RATIO = "small_value_err_ratio"
    RMSE_RATIO = "rmse_ratio"
    MAX_REL_ERR_RATIO = "max_rel_err_ratio"
    MEAN_REL_ERR_RATIO = "mean_rel_err_ratio"
    EB_RATIO = "eb_ratio"

    SMALL_VALUE = "small_value"
    RMSE = "rmse"
    MAX_REL_ERR = "max_rel_err"
    MEAN_REL_ERR = "mean_rel_err"
    EB = "eb"

    SMALL_VALUE_ERR_STATUS = "small_value_err_status"
    RMSE_STATUS = "rmse_status"
    MAX_REL_ERR_STATUS = "max_rel_err_status"
    MEAN_REL_ERR_STATUS = "mean_rel_err_status"
    EB_STATUS = "eb_status"

    MEAN_ULP_ERR = "mean_ulp_err"
    ULP_ERR_PROPORTION = "ulp_err_proportion"
    ULP_ERR_PROPORTION_RATIO = "ulp_err_proportion_ratio"

    ULP_ERR_STATUS = "ulp_err_status"

    ALL_COMPARE_INDEX = [COSINE, EUC_DIST, MAX_ABS_ERR, MAX_RELATIVE_ERR,
                         ONE_THOUSANDTH_ERR_RATIO, FIVE_THOUSANDTHS_ERR_RATIO]
    SUMMARY_COMPARE_INDEX = [MAX_DIFF, MIN_DIFF, MEAN_DIFF, NORM_DIFF,
                             MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR]
    MD5_COMPARE_INDEX = [RESULT]

    BASIC_INFO = [NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, NPU_REQ_GRAD, BENCH_REQ_GRAD]
    SUMMARY_INFO = [NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM, BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM]

    COMPARE_RESULT_HEADER = BASIC_INFO + ALL_COMPARE_INDEX + SUMMARY_INFO + [REQ_GRAD_CONSIST, RESULT, ERROR_MESSAGE]

    SUMMARY_COMPARE_RESULT_HEADER = BASIC_INFO + SUMMARY_COMPARE_INDEX + SUMMARY_INFO + [REQ_GRAD_CONSIST, RESULT,
                                                                                         ERROR_MESSAGE]

    MD5_COMPARE_RESULT_HEADER = BASIC_INFO + [NPU_MD5, BENCH_MD5, REQ_GRAD_CONSIST] + [RESULT, ERROR_MESSAGE]

    COMPARE_RESULT_HEADER_STACK = COMPARE_RESULT_HEADER + [STACK]

    SUMMARY_COMPARE_RESULT_HEADER_STACK = SUMMARY_COMPARE_RESULT_HEADER + [STACK]

    MD5_COMPARE_RESULT_HEADER_STACK = MD5_COMPARE_RESULT_HEADER + [STACK]

    HEAD_OF_COMPARE_MODE = {
        Const.ALL: COMPARE_RESULT_HEADER,
        Const.SUMMARY: SUMMARY_COMPARE_RESULT_HEADER,
        Const.MD5: MD5_COMPARE_RESULT_HEADER
    }

    # dtype match

    DTYPE_MATCH_GROUPS = [
        {Const.FLOAT16, Const.FLOAT32, Const.BFLOAT16},
        {Const.TORCH_FLOAT16, Const.TORCH_FLOAT32, Const.TORCH_BFLOAT16}
    ]

    # read_op
    IO_NAME_MAPPING = {
        Const.INPUT_ARGS: '.input',
        Const.INPUT_KWARGS: '.input',
        Const.INPUT: '.input',
        Const.OUTPUT: '.output',
        Const.PARAMS: '.parameters'
    }

    # state to struct mapping
    STATE_TO_STRUCT_MAPPING = {
        Const.INPUT: INPUT_STRUCT,
        Const.KWARGS: INPUT_STRUCT,
        Const.OUTPUT: OUTPUT_STRUCT,
        Const.PARAMS: PARAMS_STRUCT,
        Const.PARAMS_GRAD: PARAMS_GRAD_STRUCT,
        Const.DEBUG: DEBUG_STRUCT
    }

    # compare standard
    HUNDRED_RATIO_THRESHOLD = 0.01
    THOUSAND_RATIO_THRESHOLD = 0.001
    FIVE_THOUSAND_RATIO_THRESHOLD = 0.005
    TEN_THOUSAND_RATIO_THRESHOLD = 0.0001
    COSINE_THRESHOLD = 0.9999
    ULP_FLOAT32_THRESHOLD = 32
    ULP_FLOAT16_THRESHOLD = 1

    # compare result data
    NO_REAL_DATA = 'No real data'
    API_UNMATCH = 'api unmatched'
    READ_NONE = 'No data'
    NONE = 'None'
    SHAPE_UNMATCH = 'shape unmatched'
    DIFF = 'Different'
    DIFF_FLAG = 'Diff'
    UNSUPPORTED = 'unsupported'
    NAN = 'Nan'
    PASS = 'pass'
    WARNING = 'warning'
    ERROR = 'error'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    SKIP = 'SKIP'
    N_A = 'N/A'
    INF = 'inf'
    NEG_INF = '-inf'
    BFLOAT16_MIN = -3.3895313892515355e+38
    BFLOAT16_MAX = 3.3895313892515355e+38
    BFLOAT16_EPS = 3.90625e-3  # 2 ** -8

    # accuracy standards
    COS_THRESHOLD = 0.99
    MAX_ABS_ERR_THRESHOLD = 0.001
    MAX_RELATIVE_ERR_THRESHOLD = 0.001
    COS_MAX_THRESHOLD = 0.9
    MAX_ABS_ERR_MAX_THRESHOLD = 1
    ACCURACY_CHECK_YES = "Yes"
    ACCURACY_CHECK_NO = "No"
    ACCURACY_CHECK_UNMATCH = "Unmatched"

    # error message
    NO_BENCH = "No bench data matched."

    # compare const
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble]

    # acc_check const
    MAX_TOKENS = 65536
    SPECIAL_SPARSE_MOED = 4

    OVERFLOW_LIST = ['nan\t', 'inf\t', '-inf\t', 'nan', 'inf', '-inf']

    MS_GRAPH_BASE = {
        NPU_NAME: None, BENCH_NAME: None, NPU_DTYPE: None, BENCH_DTYPE: None, NPU_SHAPE: None, BENCH_SHAPE: None,
        NPU_MAX: None, NPU_MIN: None, NPU_MEAN: None, NPU_NORM: None, BENCH_MAX: None, BENCH_MIN: None,
        BENCH_MEAN: None, BENCH_NORM: None, ACCURACY: '', ERROR_MESSAGE: ''
    }
    MS_GRAPH_NPY = {
        COSINE: None, EUC_DIST: None, MAX_ABS_ERR: None, MAX_RELATIVE_ERR: None, ONE_THOUSANDTH_ERR_RATIO: None,
        FIVE_THOUSANDTHS_ERR_RATIO: None
    }
    MS_GRAPH_STATISTIC = {
        MAX_DIFF: None, MIN_DIFF: None, MEAN_DIFF: None, NORM_DIFF: None, MAX_RELATIVE_ERR: None,
        MIN_RELATIVE_ERR: None, MEAN_RELATIVE_ERR: None, NORM_RELATIVE_ERR: None
    }
    MS_GRAPH_CSV = {
        NPU_CSV_FILE: None, BENCH_CSV_FILE: None
    }

    API_MAPPING_KEYS_TO_COMPARE = [
        ('ms_args', 'pt_args'),
        ('ms_outputs', 'pt_outputs'),
        ('ms_parameters', 'pt_parameters'),
        ('ms_parameters_grad', 'pt_parameters_grad')
    ]

    INPUT_PATTERN = Const.SEP + Const.INPUT + Const.SEP
    KWARGS_PATTERN = Const.SEP + Const.KWARGS + Const.SEP
    OUTPUT_PATTERN = Const.SEP + Const.OUTPUT + Const.SEP
    PARAMS_PATTERN = Const.SEP + Const.PARAMS + Const.SEP
    PARAMS_GRAD_PATTERN = Const.SEP + Const.PARAMS_GRAD + Const.SEP

    CMP_KEY = 'compare_key'
    CMP_SHAPE = 'compare_shape'

    OP_NAME_X = 'op_name_x'
    MATCH_RESULT_COLUMNS = [
        OP_NAME_X, 'dtype_x', 'shape_x', 'summary_x', 'stack_info_x', 'state_x', 'api_origin_name_x',
        'requires_grad_x', 'data_name_x',
        CMP_KEY, CMP_SHAPE,
        'op_name_y', 'dtype_y', 'shape_y', 'summary_y', 'stack_info_y', 'state_y', 'api_origin_name_y',
        'requires_grad_y', 'data_name_y'
    ]

    INTERNAL_API_MAPPING_FILE = 'ms_to_pt_api.yaml'
    UNREADABLE = 'unreadable data'
    NPU_DUMP_DATA_DIR = 'npu_dump_data_dir'
    BENCH_DUMP_DATA_DIR = 'bench_dump_data_dir'
    NO_REAL_DATA_FLAG = '-1'

    # ATB data comparison
    STATS_COMPARISION_MODE = 'stats'
    TENSOR_COMPARISION_MODE = 'tensor'
    UNKNOWN_COMPARISION_MODE = "unknown"
    ATB_DATA_NAME = 'Data Name'
    DEVICE_AND_PID = 'Device and PID'
    EXECUTION_COUNT = 'Execution Count'
    DATA_TYPE = 'Data Type'
    DATA_SHAPE = 'Data Shape'
    TENSOR_PATH = "tensor_path"

    VERL_STOP_PARSE_RULES = {
        'fsdp': ['Qwen2Model', 'Qwen3Model', 'Qwen3MoeModel'],
        'megatron': ['GPTModel']
    }

    VERL_BEGIN_PARSE_RULES = {
        'fsdp': 'Embedding',
        'megatron': 'VocabParallelEmbedding'
    }

    BACKEND_CAT_RULES = {
        'fsdp': lambda op_name: '.q_proj.' in op_name or '.gate_proj.' in op_name,
        'megatron': lambda op_name: '.gate_up_proj.' in op_name
    }

    # 因pandas版本特性导致乱序加的排序列
    NPU_PARSE_ORDER = "npu_parse_order"
    BENCH_PARSE_ORDER = "bench_parse_order"
    COMPARE_ORDER = "compare_order"


class FileCheckConst:
    """
    Class for file check const
    """
    READ_ABLE = "r"
    WRITE_ABLE = "w"
    EXECUTE_ABLE = "x"
    READ_WRITE_ABLE = "rw"
    READ_EXECUTE_ABLE = "rx"
    WRITE_EXECUTE_ABLE = "wx"
    READ_WRITE_EXECUTE_ABLE = "rwx"
    PERM_OPTIONS = [
        READ_ABLE,
        WRITE_ABLE,
        EXECUTE_ABLE,
        READ_WRITE_ABLE,
        READ_EXECUTE_ABLE,
        WRITE_EXECUTE_ABLE,
        READ_WRITE_EXECUTE_ABLE
    ]

    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    JSON_SUFFIX = ".json"
    PT_SUFFIX = ".pt"
    CSV_SUFFIX = ".csv"
    XLSX_SUFFIX = ".xlsx"
    YAML_SUFFIX = ".yaml"
    IR_SUFFIX = ".ir"
    ZIP_SUFFIX = ".zip"
    SHELL_SUFFIX = ".sh"
    LOG_SUFFIX = ".log"
    DB_SUFFIX = '.db'
    ONNX_SUFFIX = '.onnx'
    OM_SUFFIX = '.om'
    BIN_SUFFIX = '.bin'
    MAX_BIN_SIZE = 10737418240          # 10G
    MAX_COMMON_FILE_SIZE = 10737418240  # 10G
    MAX_PKL_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_NUMPY_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_JSON_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_PT_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_XLSX_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_YAML_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_IR_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_ZIP_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_FILE_IN_ZIP_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_FILE_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    COMMON_FILE_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_LOG_SIZE = 10737418240  # 1 * 1024 * 1024 * 1024
    MAX_DB_SIZE = 107374182400  # 100 * 1024 * 1024 * 1024
    MAX_ONNX_SIZE = 10737418240  # 1 * 1024 * 1024 * 1024
    MAX_OM_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    DIR = "dir"
    FILE = "file"
    DATA_DIR_AUTHORITY = 0o750
    DATA_FILE_AUTHORITY = 0o640
    FILE_SIZE_DICT = {
        PKL_SUFFIX: MAX_PKL_SIZE,
        NUMPY_SUFFIX: MAX_NUMPY_SIZE,
        JSON_SUFFIX: MAX_JSON_SIZE,
        PT_SUFFIX: MAX_PT_SIZE,
        CSV_SUFFIX: MAX_CSV_SIZE,
        XLSX_SUFFIX: MAX_XLSX_SIZE,
        YAML_SUFFIX: MAX_YAML_SIZE,
        IR_SUFFIX: MAX_IR_SIZE,
        ZIP_SUFFIX: MAX_ZIP_SIZE,
        LOG_SUFFIX: MAX_LOG_SIZE,
        DB_SUFFIX: MAX_DB_SIZE,
        ONNX_SUFFIX: MAX_ONNX_SIZE,
        OM_SUFFIX: MAX_OM_SIZE,
        BIN_SUFFIX: MAX_BIN_SIZE
    }
    CSV_BLACK_LIST = r'^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]'


class OverflowConst:
    """
    Class for Overflow
    """
    OVERFLOW_ORIGINAL_MODE = 0
    OVERFLOW_DEBUG_MODE = 1


class MsgConst:
    """
    Class for log messages const
    """
    MSPROBE_LOG_LEVEL = "MSPROBE_LOG_LEVEL"
    LOG_LEVEL_ENUM = ["0", "1", "2", "3", "4"]
    LOG_LEVEL = ["DEBUG", "INFO", "WARNING", "ERROR"]

    class LogLevel:
        class DEBUG:
            value = 0

        class INFO:
            value = 1

        class WARNING:
            value = 2

        class ERROR:
            value = 3

    SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\u000B", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]

    NOT_CREATED_INSTANCE = "PrecisionDebugger instance is not created."


class GraphMode:
    NPY_MODE = "NPY_MODE"
    STATISTIC_MODE = "STATISTIC_MODE"
    ERROR_MODE = "ERROR_MODE"


class MonitorConst:
    """
    Class for monitor const
    """

    # monitor config set default values
    DEFAULT_GRAD_ACC_STEPS = 1
    DEFAULT_START_ITERATION = 0
    DEFAULT_START_STEP = 0
    DEFAULT_MAX_COLLECT_TIMES = 1e8
    DEFAULT_MIN_COLLECT_TIMES = 0
    DEFAULT_STEP_INTERVAL = 1

    OP_LIST = ["norm", "min", "max", "zeros", "nans", "id", "mean", "shape", "dtype"]
    MONITOR_OUTPUT_DIR = "MONITOR_OUTPUT_DIR"
    DEFAULT_MONITOR_OUTPUT_DIR = "./monitor_output"
    DATABASE = "database"
    EMAIL = "email"
    OPT_TY = ['Megatron_DistributedOptimizer', 'Megatron_Float16OptimizerWithFloat16Params']
    DEEPSPEED_OPT_TY = (
        "DeepSpeedZeroOptimizer_Stage0",
        "DeepSpeedZeroOptimizer_Stage1_or_2",
        "DeepSpeedZeroOptimizer_Stage3"
    )
    DEEPSPEED_ZERO_OPT_FILTER = "DeepSpeedZeroOptimizer"
    RULE_NAME = ['AnomalyTurbulence', 'AnomalyNan']
    L2_HOOKS = ["linear_hook", "attention_hook"]
    SA_ORDERS = ["s,b,h,d", "b,s,h,d"]

    SLICE_SIZE = 20480
    # used for name
    DOT = "."
    NAME_SEP = ":"
    INPUT_GRAD = "input_grad"
    OUTPUT_GRAD = "output_grad"
    ACTV_IN = "input"
    ACTV_OUT = "output"
    ACTVGRAD_IN = "input_grad"
    ACTVGRAD_OUT = "output_grad"
    FSDP_FLAT_SEP = "_fsdp_wrapped_module."
    # used for tasks
    ACTV = "actv"
    ACTVGRAD = "actv_grad"
    POST_GRAD = "post_grad"
    PRE_GRAD = "pre_grad"
    PRE_PARAM = "param_origin"
    POST_PARAM = "param_updated"
    ACC_GRAD = "acc_grad"
    PREFIX_POST = "post"
    PREFIX_PRE = "pre"
    EXP_AVG = "exp_avg"
    EXP_AVG_SQ = "exp_avg_sq"

    CSV_HEADER = ["vpp_stage", "name", "step"]
    CSV_HEADER_MICRO_STEP = ["vpp_stage", "name", "step", "micro_step"]
    OUTPUT_DIR_PATTERN = r"([\w-]{0,20})-rank(\d{1,5})-"
    ANOMALY_JSON = "anomaly.json"
    ANALYSE_JSON = "anomaly_analyse.json"
    TENSORBOARD = "tensorboard"
    CSV = "csv"
    API = "api"
    HEADER_NAME = 'name'
    MAX_NDIGITS = 20

    DEFAULT_STAGE = -1
    FORWARD_STAGE = 0
    BACKWARD_STAGE = 1
    OPTIMIZER_STAGE = 2
    FORWARD_KEY = [ACTV]
    BACKWARD_KEY = [ACTVGRAD, PRE_GRAD, POST_GRAD, ACC_GRAD]
    OPTIMIZER_KEY = [EXP_AVG, EXP_AVG_SQ]

    TRAIN_STAGE = {}
    for key in FORWARD_KEY:
        TRAIN_STAGE[key] = FORWARD_STAGE
    for key in BACKWARD_KEY:
        TRAIN_STAGE[key] = BACKWARD_STAGE
    for key in OPTIMIZER_KEY:
        TRAIN_STAGE[key] = OPTIMIZER_STAGE
    CSV_FILE_PATTERN = r"_(\d+)-(\d+)\.csv"
