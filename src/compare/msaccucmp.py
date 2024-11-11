# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
This class mainly involves the main function.
"""

import os
import re
import sys
import argparse
import time

from cmp_utils import log, path_check, file_utils
from cmp_utils.utils import safe_path_string, check_file_size
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.reg_manager import RegManager
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.path_check import check_others_permission
from algorithm_manager.algorithm_manager import AlgorithmManagerMain
from compare_vector import VectorComparison
from conversion.shape_format_conversion import FormatConversionMain
from dump_parse.dump_data_parser import DumpDataParser
from overflow.overflow_analyse import OverflowAnalyse
from pytorch_cmp.compare_pytorch import PytorchComparison
from vector_cmp.batch_compare import BatchCompare


MIND_STUDIO_LOGO = "[Powered by MindStudio]"


def _get_algorithm_help_info() -> str:
    """
    get algorithm help info
    :return: help info
    """
    algorithm_help_info = ['<Optional> the algorithm selection, built-in algorithm where ']
    for index, item in enumerate(ConstManager.BUILT_IN_ALGORITHM):
        algorithm_help_info.append("".join([str(index), "=", item, ", "]))
    algorithm_help_info.append('The custom algorithm uses the algorithm name. '
                               'The selection splits by ",", such as "0,MaxAbsoluteError,4,CustomAlg" or "all". '
                               'The default value is "all"')
    return "".join(algorithm_help_info)


def _get_algorithm_help_info_for_int(default_value_list: list) -> str:
    """
    get algorithm help info for int
    :param default_value_list:
    :return:
    """
    algorithm_help_info = ['<Optional> the algorithm selection, ']
    for index, item in enumerate(ConstManager.BUILT_IN_ALGORITHM):
        algorithm_help_info.append("".join([str(index), ":", item, ", "]))
    default_info = "The default value is %s." % default_value_list
    algorithm_help_info.append(default_info)
    return "".join(algorithm_help_info)


def _match_built_in_arg_value(alg_arg: str) -> bool:
    # the alg value like -alg 1 2 3 --algorithm 1 4
    if alg_arg in sys.argv:
        alg_index = sys.argv.index(alg_arg)
        re_pattern = re.compile(RegManager.BUILTIN_ALGORITHM_INDEX_PATTERN)
        # check the second value after -alg match BUILTIN_ALGORITHM_INDEX_PATTERN
        if alg_index + 2 < len(sys.argv):
            if re_pattern.match(sys.argv[alg_index + 2]):
                return True
    return False


def _add_alg_argument(compare_parser: argparse.ArgumentParser) -> None:
    if _match_built_in_arg_value('-alg') or _match_built_in_arg_value('--algorithm'):
        default_value_list = []
        for index, _ in enumerate(ConstManager.BUILT_IN_ALGORITHM):
            default_value_list.append(index)
        compare_parser.add_argument(
            '-alg', '--algorithm', dest='algorithm', type=int, nargs="+", choices=default_value_list,
            default=default_value_list, help=_get_algorithm_help_info_for_int(default_value_list))
    else:
        compare_parser.add_argument(
            '-alg', '--algorithm', dest='algorithm', default="all", type=safe_path_string,
            help=_get_algorithm_help_info())


def _add_fusion_rule_argument(compare_parser: argparse.ArgumentParser) -> None:
    compare_parser.add_argument(
        '-f', '--fusion_rule_file', dest='fusion_rule_file', default='', type=safe_path_string,
        help='<Optional> the fusion rule file path')
    compare_parser.add_argument(
        '-q', '--quant_fusion_rule_file', dest='quant_fusion_rule_file', type=safe_path_string,
        default='', help='<Optional> the quant fusion rule file path')
    compare_parser.add_argument(
        '-cf', '--close_fusion_rule_file', dest='close_fusion_rule_file', type=safe_path_string,
        default='', help='<Optional> the rule file path without fusion')


def _compare_parser(compare_parser: argparse.ArgumentParser) -> None:
    compare_parser.add_argument(
        '-m', '--my_dump_path', dest='my_dump_path', default='', type=safe_path_string, required=True,
        help='<Required> my dump path, the data compared with golden data')
    compare_parser.add_argument(
        '-g', '--golden_dump_path', dest='golden_dump_path', default='', type=safe_path_string,
        help='<Required> the golden dump path', required=True)
    _add_fusion_rule_argument(compare_parser)
    compare_parser.add_argument(
        '-out', '--output', dest='output_path', type=safe_path_string, default='', help='<Optional> the output path')
    compare_parser.add_argument(
        '-c', '--custom_script_path', dest='custom_script_path', default='', type=safe_path_string,
        help='<Optional> the user-defined script path, including format conversion and algorithm')
    compare_parser.add_argument(
        '-a', '--algorithm_options', dest='algorithm_options', default='',
        help='<Optional> the arguments for each algorithm. The format is "algorithm_name:param_name='
             'param_value". The parameter splits by ",". The algorithm splits by ";". '
             'Such as "CosineSimilarity:max=1,min=0;aa:max=1,min=0"')
    _add_alg_argument(compare_parser)
    compare_parser.add_argument(
        '-map', '--mapping', dest="mapping", action="store_true", required=False,
        help="<Optional> create mappings between my output operators and ground truth one")
    compare_parser.add_argument(
        "-overflow_detection", dest="overflow_detection", action="store_true", required=False,
        help="<Optional> Operator overflow detection, only operators of the fp16 type are supported")
    compare_parser.add_argument(
        '-r', '--range', dest="range", default=None, required=False,
        help='<Optional> compare network with the range. The format is "start,end,step". '
             '`start` means the count starts position, limited to [1, op_count], default 1.'
             '`end` means the count ends position, limited to [>=start, op_count] or -1, default -1'
             '`step` limited to [1, op_count], default 1. -r command and -s command can not be used at the same time')
    compare_parser.add_argument(
        '-s', '--select', dest="select", default=None, required=False,
        help='<Optional> compare network with the range. The format is "index_1, index_2,..." Every index should be'
             'a number in the fusion operator list -r command and -s command can not be used at the same time')
    compare_parser.add_argument(
        '-p', '--post_process', dest='post_process', choices=[0, 1], type=int, default=None,
        help='<Optional> whether to extract the compare result, only pytorch is supported.'
             '0 indicates the comparison result is not extracted, 1 indicates the comparison result is extracted')

    compare_parser.add_argument(
        '-max', '--max_cmp_size', dest='max_cmp_size', type=int, default=0,
        help='<Optional> max size of tensor array to compare')

    _add_advisor_argument(compare_parser)
    _add_version_argument(compare_parser)
    _add_argument_for_single_op(compare_parser)
    _add_ffts_argument(compare_parser)


def _add_advisor_argument(compare_parser: argparse.ArgumentParser) -> None:
    compare_parser.add_argument(
        '-advisor', dest="advisor", action="store_true", required=False, help="<optional> Enable advisor after compare"
    )


def _add_ffts_argument(compare_parser: argparse.ArgumentParser) -> None:
    compare_parser.add_argument(
        '-ffts', dest="ffts", action="store_true",
        help="<optional> Enable the comparison between ffts+ and ffts+. "
             "Direct comparison is performed without data combination")


def _add_argument_for_single_op(compare_parser: argparse.ArgumentParser) -> None:
    compare_parser.add_argument('-op', '--op_name', dest='op_name', default=None, help='<Optional> operator name')
    group = compare_parser.add_mutually_exclusive_group()
    group.add_argument(
        '-o', '--output_tensor', dest='output', default=None,
        help='<Optional> the index of output, takes effect only when the "-op" exists')
    group.add_argument(
        '-i', '--input_tensor', dest='input', default=None,
        help='<Optional> the index for input, takes effect only when the "-op" exists')
    compare_parser.add_argument(
        '--ignore_single_op_result', dest="ignore_single_op_result", action="store_true", default=False, required=False,
        help='<Optional> ignore the single operator detail result, takes effect only when the "-op" exists')
    compare_parser.add_argument(
        '-n', '--topn', dest='topn', type=int, default=ConstManager.DEFAULT_TOP_N,
        help='<Optional> the TopN for the single operator detail result, takes effect '
             'only when the "-op" exists. The value ranges from 1 to 10000. The default value is 20')
    compare_parser.add_argument(
        '-ml', '--max_line', dest='max_line', type=int, default=None,
        help='<Optional> the max line count for the single operator detail result, takes effect '
             'only when the "-op" exists. The default value is 1000000, and it should range '
             'from {} to {}'.format(ConstManager.DETAIL_LINE_COUNT_RANGE_MIN, ConstManager.DETAIL_LINE_COUNT_RANGE_MAX))


def _add_version_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '-v', '--version', dest='dump_version', choices=[1, 2], type=int, default=2,
        help='<Optional> the version of the dump file, '
             '1 means the protobuf dump file, 2 means the binary dump file, the default value is 2')


def _convert_parser(covert_parser: argparse.ArgumentParser) -> None:
    group = covert_parser.add_mutually_exclusive_group()
    covert_parser.add_argument(
        '-d', '--dump_file', dest='dump_path', default='', type=safe_path_string, required=True,
        help='<Required> the dump file path, supports one file, file list(splits by ",") and directory')
    covert_parser.add_argument('-f', '--format', dest='format', default=None, help='<Optional> the format to transfer')
    covert_parser.add_argument(
        '-s', '--shape', dest='shape', default=None,
        help='<Optional> the shape for format transfer, currently only used for FRACTAL_NZ conversion, '
             'shape format is ([0-9]+,)+[0-9]+, such as 1,3,224,224')
    group.add_argument(
        '-o', '--output_tensor', dest='output', default=None,
        help='<Optional> the index for output, takes effect only when the "-f" exists')
    group.add_argument(
        '-i', '--input_tensor', dest='input', default=None,
        help='<Optional> the index for input, takes effect only when the "-f" exists')
    covert_parser.add_argument(
        '-c', '--custom_script_path', dest='custom_script_path', default=None, type=safe_path_string,
        help='<Optional> the user-defined script path, including format conversion')
    covert_parser.add_argument('-out', '--output', dest='output_path', default='', help='<Optional> the output path')
    _add_version_argument(covert_parser)
    covert_parser.add_argument(
        '-t', '--type', dest='output_file_type', choices=['npy', 'bin', 'msnpy'],
        default='npy',
        help='<Optional> the type of the output file, '
             'npy means the output is saved as numpy format, '
             'bin means the output is saved as binary format, '
             'msnpy means the output is saved as numpy format for MindSpore, '
             'the default value is npy')


def _overflow_parser(overflow_parser: argparse.ArgumentParser) -> None:
    overflow_parser.add_argument(
        '-d', '--dump_path', dest='dump_path', default='', type=safe_path_string, required=True,
        help='<Required> the dump file path')
    overflow_parser.add_argument(
        '-out', '--output', dest='output_path', default='', type=safe_path_string, required=True,
        help='<Optional> the output path')
    overflow_parser.add_argument(
        '-n', '--top_n', dest='top_num', choices=[1, 2, 3, 4, 5], type=int, default=1,
        help='<Optional> the number of overflow ops, first n will be analyzed. the default value is 1')


def _file_compare_parser(file_compare_parser: argparse.ArgumentParser) -> None:
    file_compare_parser.add_argument(
        '-m', '--my_dump_path', dest='my_dump_path', default='', type=safe_path_string,
        help='<Required> my dump path, the data compared with golden data',
        required=True)
    file_compare_parser.add_argument(
        '-g', '--golden_dump_path', dest='golden_dump_path', default='', type=safe_path_string,
        help='<Required> the golden dump path', required=True)
    file_compare_parser.add_argument(
        '-out', '--output', dest='output_path', default='', type=safe_path_string,
        help='<Required> the output path', required=True)


def _check_argument_effect(required_arg: any, options_arg: any, options_arg_str: str, required_arg_str: str) -> None:
    if required_arg is None and options_arg is not None:
        log.print_error_log(
            'The argument %r takes effect only when the "%r" exists.' % (options_arg_str, required_arg_str))
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


def _check_single_op_argument(args: argparse.Namespace) -> None:
    if args.op_name:
        log.print_error_log('When --mapping or -map exists,the -op parameter is invalid.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if args.output or args.input:
        log.print_error_log('When --mapping or -map exists,the -i or -o parameter is invalid.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if not args.fusion_rule_file and not args.quant_fusion_rule_file:
        log.print_error_log('When --mapping or -map exists,there is need to enter the -f or -q parameter.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


def _check_dump_path_exist(dump_path_array: list) -> None:
    for item_path in dump_path_array:
        ret = path_check.check_path_valid(item_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)


def _check_file_compare_file(args: argparse.Namespace, file_type) -> None:
    for file in [args.my_dump_path, args.golden_dump_path]:
        if not file.endswith(file_type):
            log.print_error_log("[file_compare] The file %r is invalid.Only support %s file." % (file, file_type))
            raise CompareError(CompareError.MSACCUCMP_INVALID_TYPE_ERROR)
        ret = path_check.check_path_valid(file, True, False, path_type=path_check.PathType.File)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)


def _check_file_compare_out(args: argparse.Namespace) -> None:
    ret = path_check.check_output_path_valid(args.output_path, exist=True)
    if ret != CompareError.MSACCUCMP_NONE_ERROR:
        log.print_error_log('[file_compare] The -out parameter: "%r" is invalid!' % args.output_path)
        raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)


def _check_hdf5_file_valid(file_path: str) -> bool:
    """
    Check file is hdf5
    :param file_path: the file path
    :return bool
    """
    return os.path.isfile(os.path.realpath(file_path)) and file_path.endswith(".h5")


def start_compare(args: argparse.Namespace) -> int:
    """
    compare entry.
    """
    if _check_hdf5_file_valid(args.my_dump_path) and _check_hdf5_file_valid(args.golden_dump_path):
        pytorch_compare = PytorchComparison(args)
        pytorch_compare.check_arguments_valid(args)
        check_file_size(args.my_dump_path, ConstManager.ONE_HUNDRED_MB)
        check_file_size(args.golden_dump_path, ConstManager.ONE_HUNDRED_MB)
        ret = check_others_permission(args.my_dump_path)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        ret = check_others_permission(args.golden_dump_path)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        ret = pytorch_compare.compare()
        return ret

    if args.post_process is not None:
        log.print_error_log('param -p only used in pytorch session.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    if os.path.isfile(os.path.realpath(args.my_dump_path)) and os.path.isfile(os.path.realpath(args.golden_dump_path)):
        ret = check_others_permission(args.my_dump_path)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        ret = check_others_permission(args.golden_dump_path)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        compare = AlgorithmManagerMain(args)
        ret = compare.process()
    elif args.fusion_rule_file != "" and BatchCompare().check_fusion_rule_json_dir(args.fusion_rule_file):
        ret = BatchCompare().compare(args)
    else:
        args = _check_advisor_effect(args)
        compare = VectorComparison(args)
        ret = compare.compare()
    return ret


def _do_cmd() -> int:
    parser = argparse.ArgumentParser(description=MIND_STUDIO_LOGO)
    subparsers = parser.add_subparsers(help='commands')
    compare_parser = subparsers.add_parser(
        'compare', help='Compare network or single op.', description=MIND_STUDIO_LOGO
    )
    covert_parser = subparsers.add_parser(
        'convert', help='Convert my dump data to numpy data or bin data.', description=MIND_STUDIO_LOGO
    )
    overflow_parser = subparsers.add_parser(
        'overflow', help='Analyze the information of the overflow operators.', description=MIND_STUDIO_LOGO
    )
    file_compare_parser = subparsers.add_parser(
        'file_compare', help='Compare two single .npy file.', description=MIND_STUDIO_LOGO
    )

    _compare_parser(compare_parser)
    _convert_parser(covert_parser)
    _overflow_parser(overflow_parser)
    _file_compare_parser(file_compare_parser)

    args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        parser.print_help()
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    if sys.argv[1] == 'compare':
        ret = _do_compare(args)
    elif sys.argv[1] == 'convert':
        ret = _do_convert(args)
    elif sys.argv[1] == 'file_compare':
        ret = _do_file_compare(args)
    else:
        ret = _do_overflow(args)

    return ret


def _check_advisor_effect(args):
    if args.advisor and args.range is not None:
        log.print_warn_log('The argument "-advisor" takes no effect when the "-r" or "--range" exists.')
        args.advisor = False
    if args.advisor and args.select is not None:
        log.print_warn_log('The argument "-advisor" takes no effect when the "-s" or "--select" exists.')
        args.advisor = False
    if args.advisor and args.op_name is not None:
        log.print_warn_log('The argument "-advisor" takes no effect when the "-op" exists.')
        args.advisor = False
    if args.advisor:
        log.print_warn_log('The argument "-advisor" will automatically configure "-overflow_detection".')
        args.overflow_detection = True
    return args


def _check_range_effect(args: argparse.Namespace) -> None:
    if not args.fusion_rule_file and args.range is not None:
        log.print_error_log('The argument "-r" or "--range" takes effect only when the "-f" exists.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if args.op_name and args.range is not None:
        log.print_error_log('The argument "-r" or "--range" exists, the "-op" parameter is invalid.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if not args.fusion_rule_file and args.select is not None:
        log.print_error_log('The argument "-s" or "--select" takes effect only when the "-f" exists.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if args.op_name and args.select is not None:
        log.print_error_log('The argument "-s" or "--select" exists, the "-op" parameter is invalid.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if args.range and args.select is not None:
        log.print_error_log('The argument "-r" and "-s" can not be used at the same time')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if args.max_cmp_size < 0:
        log.print_error_log(
            "Please enter a valid number for max_cmp_size, the max_cmp_size should be"
            " in [0, ∞), now is %s." % args.max_cmp_size)
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


def _do_compare(args: argparse.Namespace) -> int:
    _check_range_effect(args)
    if args.mapping:
        _check_single_op_argument(args)
        compare = VectorComparison(args)
        ret = compare.compare()
    else:
        _check_argument_effect(args.op_name, args.output, '"-o" or "--output_tensor"', '-op')
        _check_argument_effect(args.op_name, args.input, '"-i" or "--input_tensor"', '-op')
        if '-n' in sys.argv or '--topn' in sys.argv:
            _check_argument_effect(args.op_name, args.topn, '"-n" or "--topn"', '-op')
        if '--ignore_single_op_result' in sys.argv:
            _check_argument_effect(args.op_name, args.ignore_single_op_result, '"--ignore_single_op_result"', '-op')
        if '-ml' in sys.argv or '--max_line' in sys.argv:
            _check_argument_effect(args.op_name, args.max_line, '"-ml" or "--max_line"', '-op')
        dump_path_array = [args.my_dump_path, args.golden_dump_path]
        _check_dump_path_exist(dump_path_array)
        ret = start_compare(args)

    return ret


def _do_convert(args: argparse.Namespace) -> int:
    _check_argument_effect(args.format, args.output, '"-o" or "--output_tensor"', '-f')
    _check_argument_effect(args.format, args.input, '"-i" or "--input_tensor"', '-f')
    _check_argument_effect(args.format, args.shape, '"-s" or "--shape"', '-f')
    _check_argument_effect(args.format, args.custom_script_path, '"-c" or "--custom_script_path"', '-f')
    abs_dump_path = os.path.abspath(args.dump_path)
    if os.path.isdir(abs_dump_path) and not os.listdir(abs_dump_path):
        log.print_error_log('The dump path is empty.')
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    if args.format:
        if os.path.isdir(abs_dump_path):
            log.print_warn_log(
                'The dump path is a directory. If the -o, -i and -s arguments exist, these arguments will be ignored.'
            )
        conversion = FormatConversionMain(args)
        ret = conversion.convert_format()
    else:
        ret = DumpDataParser(args).parse_dump_data()

    return ret


def _do_file_compare(args: argparse.Namespace) -> int:
    _check_file_compare_file(args, ConstManager.NPY_SUFFIX)
    _check_file_compare_out(args)
    args.custom_script_path = ""
    args.algorithm = ConstManager.FILE_CMP_SUPPORTED_ALGORITHM
    args.algorithm_options = ""
    compare = AlgorithmManagerMain(args)
    ret = compare.process(save_result=True)

    return ret


def _do_overflow(args: argparse.Namespace) -> int:
    overflow_analyse = OverflowAnalyse(args)
    ret = overflow_analyse.check_argument(args)
    if ret == CompareError.MSACCUCMP_NONE_ERROR:
        return overflow_analyse.analyse()
    return ret


def main() -> None:
    """
    parse argument and run command
    :return:
    """
    start = time.time()
    with file_utils.UmaskWrapper():
        try:
            ret = _do_cmd()
        except CompareError as err:
            ret = err.code
        except Exception as base_err:
            log.print_error_log(f'Basic error running {sys.argv[0]}: {base_err}')
            sys.exit(1)
    end = time.time()
    log.print_info_log('The command was completed and took %d seconds.' % (end - start))
    sys.exit(ret)


if __name__ == '__main__':
    main()
