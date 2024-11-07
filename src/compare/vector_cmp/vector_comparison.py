
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
"""
Function:
This file mainly involves vector comparison class definition.
"""
import copy
import os
import sys
import multiprocessing
import argparse
import csv
import time
import psutil

from dump_parse import dump, mapping
from vector_cmp.compare_detail import detail
from algorithm_manager.algorithm_manager import AlgorithmManager
from vector_cmp.fusion_manager import compare_result
from vector_cmp.fusion_manager.compare_rule import CompareRule
from format_manager.format_manager import FormatManager
from vector_cmp.fusion_manager.compare_fusion_op import FusionOpComparison
from vector_cmp.compare_detail.compare_detail import DetailComparison
from vector_cmp.compare_detail.compare_detail import DumpDetailComparison
from dump_parse.dump import DumpType
from cmp_utils import log, utils, utils_type, path_check
from cmp_utils.utils import safe_path_string
from cmp_utils.constant.const_manager import ConstManager
from vector_cmp.range_manager.range_manager import RangeManager
from vector_cmp.range_manager.range_mode import RangeMode
from vector_cmp.range_manager.select_mode import SelectMode
from overflow.overflow_detection import OverflowDetection
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.utils import sanitize_csv_value


class VectorComparison:
    """
    The class for vector compare
    """

    MULTI_THREAD_RESULT_COUNT = 3
    MULTI_THREAD_RETURN_CODE_INDEX = 0
    MULTI_THREAD_DUMP_MATCH_INDEX = 1
    MULTI_THREAD_COMPARE_RESULT_INDEX = 2
    MULTI_THREAD_MAX_NUM = 16

    def __init__(self: any, arguments: any = None) -> None:
        self.compare_rule = None
        self.compare_data = None
        self.detail_info = None
        self.format_manager = None
        self.args = {}
        if arguments:
            self._init_by_input_arguments(arguments)
        else:
            self._init_by_input_parse()

    @staticmethod
    def _parser_cmd(parse: any) -> None:
        parse.add_argument("-l", dest="left_dump_path", type=safe_path_string,
                           help="<Required> the left dump path, the data compared with golden data", required=True)
        parse.add_argument("-r", dest="right_dump_path", type=safe_path_string,
                           help="<Required> the right dump path, the golden data", required=True)
        parse.add_argument("-o", dest="output_path", help="<Required> output file path", type=safe_path_string,
                           required=True)
        parse.add_argument("-f", dest="fusion_json_file_path", default="", type=safe_path_string,
                           help="<Optional> fusion json file path")
        parse.add_argument("-q", dest="quant_fusion_rule_file_path", type=safe_path_string,
                           default="", help="<Optional> quant fusion rule file path")
        parse.add_argument("-d", dest="op_name", default="", help="<Optional> detail operator name", required=False)
        parse.add_argument("-t", dest="detail_type", default="output", required=False,
                           help="<Optional> detail type for operator, input or output, the default is output")
        parse.add_argument("-i", dest="detail_index", default="0",
                           help="<Optional> detail index for input or output, the default is 0", required=False)
        parse.add_argument("-csv", dest="csv", action="store_true",
                           default=False, help="<Optional> save file as csv format", required=False)
        parse.add_argument("-custom", dest="custom_path", default="", type=safe_path_string,
                           help="<Optional> user-defined path, including format conversion", required=False)
        parse.add_argument("-ffts", dest="ffts", action="store_true",
                           help="<optional> Enable the comparison between ffts+ and ffts+. "
                                "Direct comparison is performed without data combination. ")

    @staticmethod
    def _process_single_op_max_line_parameters(max_line: int) -> None:
        if max_line < ConstManager.DETAIL_LINE_COUNT_RANGE_MIN or max_line > ConstManager.DETAIL_LINE_COUNT_RANGE_MAX:
            log.print_out_of_range_error(None, '--max_line argument', max_line, '{} - {}'
                                         .format(ConstManager.DETAIL_LINE_COUNT_RANGE_MIN,
                                                 ConstManager.DETAIL_LINE_COUNT_RANGE_MAX))
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def set_output_path(self: any, output_path: str) -> None:
        """
        Set output path
        :param output_path: the output path
        """
        if os.path.islink(os.path.abspath(output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.realpath(output_path)

    def check_arguments_valid(self: any) -> None:
        """
        Check arguments valid, if invalid, throw exception
        """
        self.compare_rule.check_arguments_valid()
        exist = False
        path_type = path_check.PathType.File
        if self.detail_info:
            exist = True
            path_type = path_check.PathType.Directory
            self.detail_info.check_arguments_valid()

        ret = path_check.check_output_path_valid(self.output_path, exist, path_type)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)

        # delete old result
        if os.path.exists(self.output_path) and not self.detail_info:
            os.remove(self.output_path)
        self.compare_data.check_arguments_valid(self.compare_rule.fusion_json_file_path,
                                                self.compare_rule.quant_fusion_rule_file_path,
                                                self.compare_rule.close_fusion_rule_file_path)
        self._filter_left_dump_is_npy_overflow()
        self.format_manager.check_arguments_valid()

    def compare(self: any) -> int:
        """
        Compare for vector or detail
        """
        # 1. check arguments valid
        self.check_arguments_valid()
        # 2. parse json file
        self.compare_rule.parse_fusion_rule(self.compare_data)
        if ConstManager.RANGE_MANAGER_KEY in self.args:
            self.args.get(ConstManager.RANGE_MANAGER_KEY).check_input_valid(
                self.compare_rule.fusion_info.op_list[-1].attr.get_op_sequence())
        self.args["input_nodes"] = self.compare_rule.fusion_info.input_nodes
        # 3. do compare detail
        if self.detail_info:
            return self._compare_detail()
        # 4. do mapping
        if self.args.get("mapping"):
            return self._make_table()
        # 5. do compare vector
        return self._compare_vector()

    def _init_by_input_arguments(self, arguments) -> None:
        self.compare_rule = CompareRule(arguments.fusion_rule_file,
                                        arguments.quant_fusion_rule_file,
                                        arguments.close_fusion_rule_file)
        self.compare_data = dump.CompareData(
            os.path.realpath(arguments.my_dump_path),
            os.path.realpath(arguments.golden_dump_path),
            arguments.dump_version,
            arguments.ffts,
            arguments.fusion_rule_file)
        if arguments.op_name:
            self._process_single_op_parameters(arguments)
        else:
            self._process_output_path_parameter(arguments)
        self.args["csv"] = True
        self.format_manager = FormatManager(arguments.custom_script_path)
        self.args["algorithm_manager"] = AlgorithmManager(arguments.custom_script_path,
                                                          arguments.algorithm, arguments.algorithm_options)
        self.args["mapping"] = arguments.mapping
        self.args["overflow_detection"] = arguments.overflow_detection
        self.args["advisor"] = arguments.advisor
        self.args["input_nodes"] = []
        if arguments.range:
            self.args["range"] = arguments.range
            self.args[ConstManager.RANGE_MANAGER_KEY] = RangeMode(arguments.range)
        elif arguments.select:
            self.args["select"] = arguments.select
            self.args[ConstManager.RANGE_MANAGER_KEY] = SelectMode(arguments.select)
        self.args["my_dump_path"] = arguments.my_dump_path
        self.args["golden_dump_path"] = arguments.golden_dump_path
        self.args["max_cmp_size"] = arguments.max_cmp_size

    def _init_by_input_parse(self) -> None:
        parse = argparse.ArgumentParser()
        self._parser_cmd(parse)
        args, _ = parse.parse_known_args(sys.argv[1:])

        self.compare_rule = CompareRule(args.fusion_json_file_path,
                                        args.quant_fusion_rule_file_path)

        self.compare_data = dump.CompareData(os.path.realpath(args.left_dump_path),
                                             os.path.realpath(args.right_dump_path), ConstManager.OLD_DUMP_TYPE,
                                             args.ffts,
                                             args.fusion_json_file_path
                                             )
        if os.path.islink(os.path.abspath(args.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % args.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.realpath(args.output_path)
        if args.op_name:
            tensor_id = detail.TensorId(args.op_name, args.detail_type, args.detail_index)
            self.detail_info = detail.DetailInfo(tensor_id, ConstManager.DEFAULT_TOP_N, ignore_result=False,
                                                 max_line=ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        self.args["csv"] = args.csv
        self.format_manager = FormatManager(args.custom_path)
        self.args["algorithm_manager"] = AlgorithmManager('', 'all', '')
        self.args["mapping"] = False
        self.args["my_dump_path"] = args.left_dump_path
        self.args["golden_dump_path"] = args.right_dump_path

    def _check_both_dump_data(self: any) -> bool:
        both_dump_data = False
        if DumpType.Offline == self.compare_data.left_dump_info.type \
                and DumpType.Offline == self.compare_data.right_dump_info.type:
            both_dump_data = True
            if self.args.get("overflow_detection"):
                log.print_warn_log('Both compare data are NPU dump data, not support overflow detection.')
        return both_dump_data

    def _process_output_path_parameter(self: any, arguments: any) -> None:
        if arguments.mapping:
            file_name = 'mapping_%s.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        else:
            file_name = 'result_%s.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        if os.path.islink(os.path.abspath(arguments.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % arguments.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.join(os.path.realpath(arguments.output_path), file_name)

    def _process_single_op_parameters(self: any, arguments: any) -> None:
        tensor_type = ConstManager.OUTPUT
        tensor_index = '0'
        max_line = ConstManager.MAX_DETAIL_INFO_LINE_COUNT
        if arguments.max_line is not None:
            self._process_single_op_max_line_parameters(arguments.max_line)
            max_line = arguments.max_line
        if arguments.input:
            tensor_type = ConstManager.INPUT
            tensor_index = arguments.input
        if arguments.output:
            tensor_type = ConstManager.OUTPUT
            tensor_index = arguments.output
        if os.path.islink(os.path.abspath(arguments.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % arguments.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.realpath(arguments.output_path)
        tensor_id = detail.TensorId(arguments.op_name, tensor_type, tensor_index)
        self.detail_info = detail.DetailInfo(tensor_id, arguments.topn, arguments.ignore_single_op_result, max_line)

    def _filter_left_dump_is_npy_overflow(self) -> None:
        """
        npy doesn't support overflow detection.
        We need turned the parameter 'overflow_detection' to False.
        Different types of dump files have different naming rules.
        The suffix of an npy file is '.npy'. Therefore, you only need to
        check the suffix of the path corresponding to any operator to
        determine whether the file is a npy file.
        If the parameter 'overflow_detection' is set to True, change it to False.
        """
        if self.args.get('overflow_detection') and self.compare_data.left_dump_info.op_name_to_file_map:
            file_path_list = []
            for _, file_path in self.compare_data.left_dump_info.op_name_to_file_map.items():
                file_path_list = file_path
                if file_path_list:
                    break
            if file_path_list:
                file_name = file_path_list[0]
                if file_name.endswith('.npy'):
                    self.args['overflow_detection'] = False

    def _compare_fusion_ops(self: any, fusion_op_names: list, lock: any) -> list:
        all_cmp_res = []

        _ = lock  # Bypassing parameter lock is not used
        for op_name in fusion_op_names:
            res = self._compare_by_fusion_op(op_name)
            all_cmp_res.append(res)
            # save result when 1000 operators are compared
        return all_cmp_res

    def _get_result_list(self, res):
        result = []
        for single_op_result in res[self.MULTI_THREAD_COMPARE_RESULT_INDEX]:
            for item in single_op_result.result_list:
                result.append(item)
        return result

    def _write_result_to_writer(self: any, result: list, output_file: any) -> None:
        for res in result:
            if len(res) != self.MULTI_THREAD_RESULT_COUNT:
                continue
            item_list = self._get_result_list(res)
            for item in item_list:
                if self.args.get("csv"):
                    writer = csv.writer(output_file)
                    sanitized_item = [sanitize_csv_value(cell) for cell in item]
                    writer.writerow(sanitized_item)
                else:
                    item.pop()
                    each_row_str = " ".join(item)
                    output_file.write(f"\n{each_row_str}")

    def _save_cmp_result(self: any, result: list, lock: any) -> None:
        lock.acquire()
        try:
            with os.fdopen(os.open(self.output_path, ConstManager.WRITE_FLAGS,
                                   ConstManager.WRITE_MODES), 'a+', newline='') as output_file:
                self._write_result_to_writer(result, output_file)
        except IOError as io_error:
            log.print_open_file_error(self.output_path, io_error)
        finally:
            lock.release()

    def _compare_by_fusion_op(self: any, fusion_op_name: str) -> (int, bool, list):
        comparison = FusionOpComparison(fusion_op_name, self.compare_rule, self.compare_data, self.format_manager,
                                        self.args)
        return comparison.compare()

    def _get_max_process_num(self) -> int:
        if self.MULTI_THREAD_MAX_NUM == 1:
            return 1  # Bypassing test code entering `os.listdir`

        golden_dump_path = self.args.get("golden_dump_path")
        file_sizes = [1] + [os.path.getsize(os.path.join(golden_dump_path, ii)) for ii in os.listdir(golden_dump_path)]
        max_file_size = max(file_sizes)

        mem = psutil.virtual_memory()
        available_mem = mem.available
        mem_max_process_num = available_mem // max_file_size // 4

        cpu_max_process_num = int((multiprocessing.cpu_count() + 1) / 2)
        return min(mem_max_process_num, cpu_max_process_num, self.MULTI_THREAD_MAX_NUM)

    def _handle_multi_process(self: any, func: any, lock: any = None) -> list:
        # 2. compare operator by multi processes
        # 1 ensure multi processes number, which is half of the CPUs
        process_num = self._get_max_process_num()
        # 2 get all operator names
        if ConstManager.RANGE_MANAGER_KEY in self.args:
            all_op_names = self.args.get(ConstManager.RANGE_MANAGER_KEY).get_all_ops(self.compare_rule)
        else:
            all_op_names = self.compare_rule.fusion_info.fusion_op_name_to_op_map.keys()
        # 3 allocate all operator names evenly by multi processes number
        op_names = []
        for _ in range(process_num):
            op_names.append([])
        for i, op_name in enumerate(all_op_names):
            op_names[i % process_num].append(op_name)
        # 4 start multi processes, then waiting subprocess end of running
        all_task = []
        pool = multiprocessing.Pool(process_num)
        for fusion_op_names in op_names:
            if lock:
                task = pool.apply_async(func, args=(fusion_op_names, lock))
            else:
                task = pool.apply_async(func, args=(fusion_op_names,))
            all_task.append(task)
        pool.close()
        pool.join()
        return all_task

    def _compare_by_multi_process(self: any) -> (int, bool):
        # 1. write header to file
        if not self._write_header_to_file():
            return CompareError.MSACCUCMP_OPEN_FILE_ERROR, False
        # 2. compare operator by multi-processing
        all_task = self._handle_multi_process(self._compare_fusion_ops, multiprocessing.Manager().RLock())

        # 3. check subprocess return value
        ret = CompareError.MSACCUCMP_NONE_ERROR
        dump_match = False
        all_result = []
        result_mapping = {}
        cmp_res = []

        for task in all_task:
            all_result.extend(task.get())
        for res in all_result:
            if len(res) != self.MULTI_THREAD_RESULT_COUNT:
                continue
            if not ret:
                ret = res[self.MULTI_THREAD_RETURN_CODE_INDEX]
            if not dump_match:
                dump_match = res[self.MULTI_THREAD_DUMP_MATCH_INDEX]
            for single_op_cmp_res in res[self.MULTI_THREAD_COMPARE_RESULT_INDEX]:
                if single_op_cmp_res.is_ffts and not single_op_cmp_res.npu_vs_npu:
                    result_mapping[single_op_cmp_res.op_name] = single_op_cmp_res

        for i, res in enumerate(all_result):
            for single_op_cmp_res in res[self.MULTI_THREAD_COMPARE_RESULT_INDEX]:
                if single_op_cmp_res.is_ffts and single_op_cmp_res.op_name_origin_output_index_map:
                    single_op_cmp_res.find_pre_op(result_mapping)

            cmp_res.append(res)
            if (i + 1) % 1000 == 0:
                self._save_cmp_result(cmp_res, multiprocessing.Manager().RLock())
                cmp_res.clear()
        self._save_cmp_result(cmp_res, multiprocessing.Manager().RLock())
        return ret, dump_match

    def _compare_vector(self: any) -> int:
        ret, dump_match = self._compare_by_multi_process()
        utils.sort_result_file_by_index(self.output_path, self.args.get("csv"))
        if not dump_match:
            ret = CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR
            if self.args.get("range"):
                log.print_warn_log('The model in [%s] range does not match the dump data.' % self.args.get('range'))
            elif self.args.get("select"):
                log.print_warn_log('The model in index list [%s] does not match '
                                   'the dump data.' % self.args.get('select'))
            else:
                log.print_error_log('The model does not match the dump data, '
                                    'please check the model and the dump data again.')
        else:
            if os.path.exists(self.output_path):
                log.print_write_result_info('comparison result', self.output_path)
                if self.args.get("advisor"):
                    self._do_advisor()
        return ret

    def _do_advisor(self):
        try:
            from advisor.compare_advisor import CompareAdvisor
        except ImportError as import_error:
            log.print_warn_log("Unable to import module: %s." % str(import_error))
            log.print_warn_log("Skip compare results Analysis.")
        else:
            out_path = os.path.dirname(self.output_path)
            compare_advisor = CompareAdvisor(self.output_path, self.args.get("input_nodes"), out_path)
            advisor_result = compare_advisor.advisor()
            message_list = advisor_result.print_advisor_log()
            advisor_result.gen_summary_file(out_path, message_list)

    def _compare_detail(self: any) -> int:
        """
        Compare detail by op name
        :return VectorComparisonErrorCode
        """
        if self.compare_rule.fusion_json_file_path == "" and self.compare_rule.quant_fusion_rule_file_path == "":
            log.print_warn_log('Both the offline fusion rule file path and '
                               'the quant fusion rule file path cannot be empty. '
                               'Please ensure that the data is reasonable.')
            if self.args.get("overflow_detection"):
                log.print_warn_log('Both compare data are NPU dump data, not support overflow detection.')
            comparison = DumpDetailComparison(self.detail_info, self.compare_data, self.output_path)
            return comparison.compare()
        if self.detail_info.tensor_id.op_name not in self.compare_rule.fusion_info.op_name_to_fusion_op_name_map:
            log.print_error_log('There is no "%s" in the fusion rule file.' % self.detail_info.tensor_id.op_name)
            return CompareError.MSACCUCMP_INVALID_PARAM_ERROR
        if self.args.get("overflow_detection") and not self._check_both_dump_data():
            overflow_detection = OverflowDetection(self.compare_data, self.detail_info.tensor_id.op_name)
            overflow_detection.process_op_overflow_detection()
        fusion_op_name = self.compare_rule.fusion_info.op_name_to_fusion_op_name_map.get(
            self.detail_info.tensor_id.op_name)
        fusion_op_comparison = FusionOpComparison(fusion_op_name, self.compare_rule, self.compare_data,
                                                  self.format_manager, self.args)
        comparison = DetailComparison(self.detail_info, fusion_op_comparison, self.output_path)
        return comparison.compare()

    def _write_header_to_file(self: any) -> bool:
        cur_op_header = self._pre_handle_header()
        try:
            with os.fdopen(os.open(self.output_path, ConstManager.WRITE_FLAGS,
                                   ConstManager.WRITE_MODES), 'a+', newline='') as output_file:
                header = compare_result.get_result_title(self.args.get('algorithm_manager'), cur_op_header,
                                                         self.args.get('overflow_detection'))
                if self.args.get("csv"):
                    writer = csv.writer(output_file)
                    sanitized_header = [sanitize_csv_value(cell) for cell in header]
                    writer.writerow(sanitized_header)
                else:
                    output_file.write(" ".join(header))
        except IOError as io_error:
            log.print_open_file_error(self.output_path, io_error)
            return False
        return True

    def _make_mapping_table_by_op_name(self: any, fusion_op_names: list) -> list:
        all_cmp_res = []
        for op_name in fusion_op_names:
            res = FusionOpComparison(op_name, self.compare_rule, self.compare_data, self.format_manager,
                                     self.args).make_gpu_and_npu_mapping_table()
            all_cmp_res += res
        return all_cmp_res

    def _make_table(self: any) -> int:
        all_task = self._handle_multi_process(self._make_mapping_table_by_op_name)
        origin_list = []
        for task in all_task:
            origin_list += task.get()
        origin_list.sort(key=lambda xx: int(xx[0]))
        try:
            with os.fdopen(os.open(self.output_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES), 'a+',
                           newline='') as out_file:
                writer = csv.writer(out_file)
                header = ConstManager.MAPPING_FILE_HEADER
                RangeManager.adjust_header(header)
                sanitized_header = [sanitize_csv_value(cell) for cell in header]
                writer.writerow(sanitized_header)
                for item in origin_list:
                    sanitized_item = [sanitize_csv_value(cell) for cell in item]
                    writer.writerow(sanitized_item)
        except IOError as io_error:
            log.print_open_file_error(self.output_path, io_error)
            raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from io_error
        if os.path.exists(self.output_path):
            log.print_write_result_info('mapping table result', self.output_path)
        return CompareError.MSACCUCMP_NONE_ERROR

    def _pre_handle_header(self: any) -> list:
        op_header = copy.deepcopy(ConstManager.VECTOR_COMPARE_HEADER)
        golden_dump_path = self.args.get("golden_dump_path")
        my_dump_path = self.args.get("my_dump_path")
        address_index = [i for i, x in enumerate(op_header) if x == 'Address']
        if utils.dump_path_contains_npy(golden_dump_path) and len(address_index) > 0:
            op_header.pop(address_index[-1])
        if utils.dump_path_contains_npy(my_dump_path) and len(address_index) > 0:
            op_header.pop(address_index[0])
        return op_header

