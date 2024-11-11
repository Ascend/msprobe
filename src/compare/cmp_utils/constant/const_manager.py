
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the const value.
"""
import os
import stat
import numpy as np

import dump_data_pb2 as DD


class ConstManager:
    """
    The class for const manager
    """
    # common const
    LINUX_FILE_NAME_MAX_LEN = 255
    INPUT = 'input'
    OUTPUT = 'output'
    SUPPORT_DETAIL_TYPE = [INPUT, OUTPUT]
    FOUR_DIMS_LENGTH = 4
    FIVE_DIMS_LENGTH = 5
    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
    OLD_DUMP_TYPE = 0
    PROTOBUF_DUMP_TYPE = 1
    BINARY_DUMP_TYPE = 2
    N0_AXIS = 16
    C0_AXIS = 16

    TIMESTAMP_LENGTH = 16
    INVALID_TIMESTAMP = 0
    INVALID_FILE_TYPE = 0
    TRAD_MODE = 0


    # overflow
    OVERFLOW_MIN_VALUE = 5.96e-8
    OVERFLOW_MAX_VALUE = 65504
    MAGIC_KEY_WORD = 'magic'
    MAGIC_NUM = 0x5a5a5a5a
    INVALID_ID = 65535

    # algorithm
    FLOAT_EPSILON = np.finfo(float).eps
    NAN = 'NaN'

    # progress const
    MAX_PROGRESS_LINE_COUNT = 100000
    MAX_PROGRESS = 100

    TIME_LENGTH = 1000
    DELIMITER = ':'

    # special ops
    SPECIAL_OPS_TYPE = ("Cast", "TransData")

    # single op const

    DEFAULT_TOP_N = 20
    MAX_DETAIL_INFO_LINE_COUNT = 1000000
    DETAIL_LINE_COUNT_RANGE_MAX = 1000000
    DETAIL_LINE_COUNT_RANGE_MIN = 10000
    CSV_SUFFIX = ".csv"
    TXT_SUFFIX = ".txt"
    NPY_SUFFIX = '.npy'
    SUMMARY_TXT_SUFFIX = "_summary.txt"
    SIMPLE_OP_MAPPING_FILE_NAME = "simple_op_mapping.csv"

    FUSION_OP_INDEX = 1

    # algorithm manager const
    BUILT_IN_ALGORITHM_DIR_NAME = "builtin_algorithm"
    CUSTOM_ALGORITHM_DIR_NAME = "algorithm"
    COMPARE_FUNC_NAME = 'compare'
    BUILT_IN_ALGORITHM = [
        "CosineSimilarity",
        "MaxAbsoluteError",
        "AccumulatedRelativeError",
        "RelativeEuclideanDistance",
        "KullbackLeiblerDivergence",
        "StandardDeviation",
        "MeanAbsoluteError",
        "RootMeanSquareError",
        "MaxRelativeError",
        "MeanRelativeError"
    ]
    ALGORITHM_FILE_NAME_PATTERN = r"^alg_([A-Za-z0-9_]+)\.py[c]?$"
    BOOL_ALGORITHM = ('CosineSimilarity', 'RelativeEuclideanDistance')
    FILE_CMP_SUPPORTED_ALGORITHM = [
        "CosineSimilarity",
        "MaxAbsoluteError",
        "AccumulatedRelativeError",
        "RelativeEuclideanDistance",
        "MeanAbsoluteError",
        "RootMeanSquareError",
        "MaxRelativeError",
        "MeanRelativeError"
    ]

    COMPARE_ARGUMENT_COUNT = 3
    SPACE = ' '
    BUILTIN = 'Builtin'
    CUSTOM = 'Custom'

    # built in algorithm const
    MINIMUM_VALUE = 0.001

    INPUT_OP_NAME_INDEX = 0
    INPUT_INDEX_INDEX = 1
    INPUT_INFO_COUNT = 2

    TENSOR_INDEX = 7
    INPUT_PATTERN = ":input:"
    OUTPUT_PATTERN = ":output:"

    # fusion rule const
    GRAPH_OBJECT = "graph"
    OP_OBJECT = "op"
    NAME_OBJECT = "name"
    TYPE_OBJECT = "type"
    ID_OBJECT = 'id'
    INPUT_OBJECT = "input"
    ATTR_OBJECT = "attr"
    L1_FUSION_SUB_GRAPH_NO_OBJECT = "_L1_fusion_sub_graph_no"
    ORIGINAL_OP_NAMES_OBJECT = "_datadump_original_op_names"
    OUTPUT_DESC_OBJECT = "output_desc"
    ORIGIN_NAME_OBJECT = "_datadump_origin_name"
    ORIGIN_OUTPUT_INDEX_OBJECT = "_datadump_origin_output_index"
    ORIGIN_FORMAT_OBJECT = "_datadump_origin_format"
    IS_MULTI_OP = "_datadump_is_multiop"
    GE_ORIGIN_FORMAT_OBJECT = "origin_format"
    GE_ORIGIN_SHAPE_OBJECT = "origin_shape"
    D_TYPE = "dtype"
    KEY_OBJECT = "key"
    VALUE_OBJECT = "value"
    STRING_TYPE_OBJECT = "s"
    INT_TYPE_OBJECT = "i"
    BOOL_TYPE_OBJECT = 'b'
    LIST_TYPE_OBJECT = "list"
    DATA_OBJECT = "Data"

    # network compare range
    START_INDEX = 1
    END_INDEX = 1
    STEP_INDEX = 1
    DEFAULT_START = 1
    DEFAULT_END = -1
    DEFAULT_STEP = 1
    OP_SEQUENCE_INDEX = 1
    OP_SEQUENCE = 'OpSequence'
    RANGE_MANAGER_KEY = 'range_manager'
    RANGE_MODE = 'range'
    SELECT_MODE = 'select'

    # dump file
    LEFT_TYPE = 'Left'
    RIGHT_TYPE = 'Right'

    VECTOR_COMPARE_HEADER = [
        "OpType", "NPUDump", "DataType", "Address",
        "GroundTruth", "DataType", "Address", "TensorIndex"
    ]
    MY_OUTPUT_ADDRESS_INDEX = 3
    GROUND_TRUTH_ADDRESS_INDEX = 6

    MAPPING_FILE_HEADER = [
        "Index", "OpType", "NPUDump", "GroundTruth", "TensorIndex", "NPUDumpPath", "GroundTruthPath"
    ]

    STRING_TO_FORMAT_MAP = {
        "NCHW": DD.FORMAT_NCHW,
        "NHWC": DD.FORMAT_NHWC,
        "ND": DD.FORMAT_ND,
        "NC1HWC0": DD.FORMAT_NC1HWC0,
        "FRACTAL_Z": DD.FORMAT_FRACTAL_Z,
        "NC1C0HWPAD": DD.FORMAT_NC1C0HWPAD,
        "NHWC1C0": DD.FORMAT_NHWC1C0,
        "FSR_NCHW": DD.FORMAT_FSR_NCHW,
        "FRACTAL_DECONV": DD.FORMAT_FRACTAL_DECONV,
        "C1HWNC0": DD.FORMAT_C1HWNC0,
        "FRACTAL_DECONV_TRANSPOSE": DD.FORMAT_FRACTAL_DECONV_TRANSPOSE,
        "FRACTAL_DECONV_SP_STRIDE_TRANS": DD.FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
        "NC1HWC0_C04": DD.FORMAT_NC1HWC0_C04,
        "FRACTAL_Z_C04": DD.FORMAT_FRACTAL_Z_C04,
        "CHWN": DD.FORMAT_CHWN,
        "DECONV_SP_STRIDE8_TRANS": DD.FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
        "NC1KHKWHWC0": DD.FORMAT_NC1KHKWHWC0,
        "BN_WEIGHT": DD.FORMAT_BN_WEIGHT,
        "FILTER_HWCK": DD.FORMAT_FILTER_HWCK,
        "HWCN": DD.FORMAT_HWCN,
        "LOOKUP_LOOKUPS": DD.FORMAT_HASHTABLE_LOOKUP_LOOKUPS,
        "LOOKUP_KEYS": DD.FORMAT_HASHTABLE_LOOKUP_KEYS,
        "LOOKUP_VALUE": DD.FORMAT_HASHTABLE_LOOKUP_VALUE,
        "LOOKUP_OUTPUT": DD.FORMAT_HASHTABLE_LOOKUP_OUTPUT,
        "LOOKUP_HITS": DD.FORMAT_HASHTABLE_LOOKUP_HITS,
        "MD": DD.FORMAT_MD,
        "NDHWC": DD.FORMAT_NDHWC,
        "C1HWNCoC0": DD.FORMAT_C1HWNCoC0,
        "FRACTAL_NZ": DD.FORMAT_FRACTAL_NZ,
        "NCDHW": DD.FORMAT_NCDHW,
        "DHWCN": DD.FORMAT_DHWCN,
        "NDC1HWC0": DD.FORMAT_NDC1HWC0,
        "FRACTAL_Z_3D": DD.FORMAT_FRACTAL_Z_3D,
        "CN": DD.FORMAT_CN,
        "DHWNC": DD.FORMAT_DHWNC,
        "FRACTAL_Z_3D_TRANSPOSE": DD.FORMAT_FRACTAL_Z_3D_TRANSPOSE,
        "FRACTAL_ZN_LSTM": DD.FORMAT_FRACTAL_ZN_LSTM,
        "FRACTAL_Z_G": DD.FORMAT_FRACTAL_Z_G,
        "RESERVED": DD.FORMAT_RESERVED,
        "ALL": DD.FORMAT_ALL,
        "NULL": DD.FORMAT_NULL,
        "ND_RNN_BIAS": DD.FORMAT_ND_RNN_BIAS,
        "FRACTAL_ZN_RNN": DD.FORMAT_FRACTAL_ZN_RNN,
        "YUV": DD.FORMAT_YUV,
        "YUV_A": DD.FORMAT_YUV_A,
        "MAX": DD.FORMAT_MAX,
    }

    DTYPE_KEY = 'dtype'
    STRUCT_FORMAT_KEY = 'struct_format'
    DATA_TYPE_TO_DTYPE_MAP = {
        DD.DT_FLOAT: {DTYPE_KEY: np.float32, STRUCT_FORMAT_KEY: 'f'},
        DD.DT_FLOAT16: {DTYPE_KEY: np.float16, STRUCT_FORMAT_KEY: 'e'},
        DD.DT_DOUBLE: {DTYPE_KEY: np.float64, STRUCT_FORMAT_KEY: 'd'},
        DD.DT_INT8: {DTYPE_KEY: np.int8, STRUCT_FORMAT_KEY: 'b'},
        DD.DT_INT16: {DTYPE_KEY: np.int16, STRUCT_FORMAT_KEY: 'h'},
        DD.DT_INT32: {DTYPE_KEY: np.int32, STRUCT_FORMAT_KEY: 'i'},
        DD.DT_INT64: {DTYPE_KEY: np.int64, STRUCT_FORMAT_KEY: 'q'},
        DD.DT_UINT8: {DTYPE_KEY: np.uint8, STRUCT_FORMAT_KEY: 'B'},
        DD.DT_UINT16: {DTYPE_KEY: np.uint16, STRUCT_FORMAT_KEY: 'H'},
        DD.DT_UINT32: {DTYPE_KEY: np.uint32, STRUCT_FORMAT_KEY: 'I'},
        DD.DT_UINT64: {DTYPE_KEY: np.uint64, STRUCT_FORMAT_KEY: 'Q'},
        DD.DT_BOOL: {DTYPE_KEY: np.bool_, STRUCT_FORMAT_KEY: '?'},
        DD.DT_COMPLEX64: {DTYPE_KEY: np.complex64, STRUCT_FORMAT_KEY: '?'},
        DD.DT_COMPLEX128: {DTYPE_KEY: np.complex128, STRUCT_FORMAT_KEY: '?'},
        DD.DT_BF16: {DTYPE_KEY: 'bfloat16', STRUCT_FORMAT_KEY: 'e'},
        DD.DT_UINT1: {DTYPE_KEY: np.uint8, STRUCT_FORMAT_KEY: '?'},
        DD.DT_UNDEFINED: {DTYPE_KEY: np.int8, STRUCT_FORMAT_KEY: 'b'},
        DD.DT_RESOURCE: {DTYPE_KEY: np.int64, STRUCT_FORMAT_KEY: 'q'},
    }
    DATA_TYPE_TO_STR_DTYPE_MAP = {
        DD.DT_FLOAT: "float32",
        DD.DT_FLOAT16: "float16",
        DD.DT_DOUBLE: "float64",
        DD.DT_INT8: "int8",
        DD.DT_INT16: "int16",
        DD.DT_INT32: "int32",
        DD.DT_INT64: "int64",
        DD.DT_UINT8: "uint8",
        DD.DT_UINT16: "uint16",
        DD.DT_UINT32: "uint32",
        DD.DT_UINT64: "uint64",
        DD.DT_BOOL: "bool",
        DD.DT_COMPLEX64: "complex64",
        DD.DT_COMPLEX128: "complex128",
        DD.DT_BF16: "bfloat16",
        DD.DT_UINT1: "uint1",
        DD.DT_UNDEFINED: "int8",
        DD.DT_RESOURCE: "int64",
    }

    UNPACK_DTYPE = [
        DD.DT_UINT1,
    ]
    CAST_FP32_DTYPE = [
        DD.DT_BF16,
    ]

    # Standard
    STANDARD_SUFFIX = ".pb"
    STANDARD_FILE_NAME = 'op_name.output_index.timestamp.pb'

    # Qunat
    QUANT_SUFFIX = ".quant"
    QUANT_FILE_NAME = 'op_name.output_index.timestamp.quant'

    # Offline
    OFFLINE_FILE_NAME = 'op_type.op_name.task_id(.stream_id).timestamp'
    OFFLINE_FILE_NAME_COUNT = 5

    # Standard
    NUMPY_SUFFIX = ".npy"
    NUMPY_FILE_NAME = 'op_name.output_index.timestamp.npy'

    QUANT_OP_NANE_SUFFIX_LIST = [
        "_quant_layer", "_anti_quant_layer", "AscendQuant", "AscendWeightQuant",
        "AntiQuant", "_quant", "_weight_quant", "_anti_quant"
    ]

    DEQUANT_OP_NANE_SUFFIX_LIST = ["_dequant_layer", "AscendDequant", "_dequant"]

    CHAR_FMT = 's'
    INT32_FMT = 'i'
    UINT32_FMT = 'I'
    INT64_FMT = 'q'
    UINT64_FMT = 'Q'

    CHAR_SIZE = 1
    INT16_SIZE = 2
    UINT16_SIZE = 2
    INT32_SIZE = 4
    INT64_SIZE = 8
    UINT32_SIZE = 4
    UINT64_SIZE = 8

    UNPACK_FORMAT = {
        'CHAR':   {'FMT': CHAR_FMT, 'SIZE': CHAR_SIZE},
        'UINT32': {'FMT': UINT32_FMT, 'SIZE': UINT32_SIZE},
        'UINT64': {'FMT': UINT64_FMT, 'SIZE': UINT64_SIZE},
        'INT32': {'FMT': INT32_FMT, 'SIZE': INT32_SIZE},
        'INT64': {'FMT': INT64_FMT, 'SIZE': INT64_SIZE},
    }
    ONE_GB = 1 * 1024 * 1024 * 1024
    ONE_MB = 1 * 1024 * 1024
    ONE_HUNDRED_MB = 100 * 1024 * 1024
    DHA_ATOMIC_ADD_INFO_SIZE = 128
    L2_ATOMIC_ADD_INFO_SIZE = 128
    AI_CORE_INFO_SIZE = 256
    DHA_ATOMIC_ADD_STATUS_SIZE = 256
    L2_ATOMIC_ADD_STATUS_SIZE = 256
    AI_CORE_STATUS_SIZE = 1024
    OVERFLOW_CHECK_SIZE = \
        DHA_ATOMIC_ADD_INFO_SIZE + L2_ATOMIC_ADD_INFO_SIZE + AI_CORE_INFO_SIZE + DHA_ATOMIC_ADD_STATUS_SIZE \
        + L2_ATOMIC_ADD_STATUS_SIZE + AI_CORE_STATUS_SIZE
    ACC_TYPE = {
        0: "AIC",
        1: "AIV",
        2: "AICPU",
        3: "SDMA"
    }
    OVERFLOW_DEBUG = ('magic', 'version', 'acc_list')
    ACC_DEBUG = ('valid', 'acc_type', 'rsv', 'data_len', 'data')
    AIC_AIV_DEBUG = (
        'model_id', 'stream_id', 'task_id', 'task_type', 'context_id',
        'thread_id', 'pc_start', 'para_base', 'core_id', 'block_id', 'status'
    )
    SDMA_DEBUG = (
        'model_id', 'stream_id', 'task_id', 'task_type', 'context_id',
        'thread_id', 'src_addr', 'dst_addr', 'channel_id', 'status'
    )
    AICPU_DEBUG = (
        'model_id', 'stream_id', 'task_id', 'task_type', 'context_id', 'cpu_id', 'thread_id', 'status'
    )
    DEBUG_INFO_MAP = {
        "AIC": AIC_AIV_DEBUG,
        "AIV": AIC_AIV_DEBUG,
        "SDMA": SDMA_DEBUG,
        "AICPU": AICPU_DEBUG,
    }
    HEX_FORMAT_ITEM = ("pc_start", "para_base", "src_addr", "dst_addr")

    BUFFER_TYPE_MAP = {DD.L1: 'l1'}
    CONVERT_FAILED_FILE_LIST_NAME = "convert_failed_file_list.txt"
    MAPPING_FILE_NAME = "mapping.csv"
    END_FLAG = "\0"

    AICORE = "AICORE"
    AICPU = "AICPU"
    DEBUG = "DEBUG"
    HCCL = "HCCL"
    FFTSPLUS = "FFTSPLUS"

    TASK_TYPE_MAP = {
        AICORE: '0',
        AICPU: '1',
        DEBUG: '2',
        HCCL: '3',
        FFTSPLUS: '4'
    }
    # task mode
    NORMAL_MODE = 0
    AUTOMATIC_MODE = 1
    MANUAL_MODE = 2
    SPEC_MODE = 3

    FFTS_TIMESTAMP = 'timestamp'

    INVALID_SORT_MODE = 1
    INVALID_THREAD_ID = 2
    INVALID_SLICE_X = 3


    OLD_FILE_FIELD_NUM = 4
    NEW_FILE_FIELD_NUM = 9
    # FFTS/FFTS+ MODE Field
    FFTS_MANUAL_MODE_FIELD = "_lxslice"
    SGT_FIELD = "_sgt_graph"
    LXSLICE_FILED = "lxslice"

    # walk limit
    MAX_WALK_FILE_NUM = 1000
    MAX_WALK_DIR_DEEP_NUM = 50

    @property
    def max_top_n(self: any) -> int:
        """
        max top n
        """
        return 10000

    @property
    def min_top_n(self: any) -> int:
        """
        mix top n
        """
        return 1
