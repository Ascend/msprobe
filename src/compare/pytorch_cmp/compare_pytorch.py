
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
This file mainly involves the compare_npu_vs_npu function.
"""
import argparse
import multiprocessing
import os
import csv
import time
import numpy as np

from cmp_utils import utils, utils_type, path_check, log
from pytorch_cmp import pytorch_dump_data as pytorch_dump
from algorithm_manager.algorithm_manager import AlgorithmManager
from cmp_utils.constant.const_manager import ConstManager
from vector_cmp.fusion_manager import compare_result
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.utils import sanitize_csv_value


class PytorchComparison:
    """
    The class for comparing PyTorch dump data.
    """
    OP_HEADER = ["NPUDump", "NPUDumpPath", "GroundTruthPath", "DataType"]

    def __init__(self: any, args: argparse.Namespace) -> None:
        self.compare_data = pytorch_dump.CompareData(
            os.path.realpath(args.my_dump_path),
            os.path.realpath(args.golden_dump_path))
        file_name = 'result_%s.csv' \
                    % time.strftime("%Y%m%d%H%M%S",
                                    time.localtime(time.time()))
        ret = path_check.check_output_path_valid(args.output_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        if os.path.islink(os.path.abspath(args.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % args.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.join(
            os.path.realpath(args.output_path), file_name)
        self.algorithm_manager = AlgorithmManager(args.custom_script_path,
                                                  args.algorithm,
                                                  args.algorithm_options)
        self.is_open_advisor = args.advisor
        self.filter_flag = args.post_process

    @staticmethod
    def _save_numpy_data(file_path: str, data: any) -> None:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), mode=0o700)
        np.save(file_path, data)

    @staticmethod
    def _get_item_location(row: list) -> list:
        cos_index = 0
        my_dump_index = 0
        golden_index = 0
        for (index, item) in enumerate(row):
            if item == "CosineSimilarity":
                cos_index = index
            elif item == "MyDumpDataPath":
                my_dump_index = index
            elif item == "GoldenDumpDataPath":
                golden_index = index
        return [cos_index, my_dump_index, golden_index]

    def record_not_matched(self: any) -> None:
        """
        Find all datasets that do not match my dump in the golden dump and record
        them in the result file.
        """
        not_matched_result = []
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager)
        for order, op_name, golden_dataset, message in \
                self.compare_data.get_not_matched_golden_datasets():
            not_matched_result += fusion_op_result.get_pytorch_result(
                compare_result.PytorchOpInfo(order, op_name, ConstManager.NAN, golden_dataset), None, [message])
        self._save_cmp_result(not_matched_result)

    def compare(self: any) -> int:
        """
        Compare for pytorch dump data.
        """
        log.print_info_log("start parse dump file!")
        # Write the result header
        if not self._write_header_to_file():
            return CompareError.MSACCUCMP_OPEN_FILE_ERROR

        # parse and compare
        try:
            ret = self.compare_data.parse_dump_file()
        except CompareError as error:
            return error.code
        finally:
            self.compare_data.close_file()

        # compare dump file
        if ret == CompareError.MSACCUCMP_NONE_ERROR:
            return self._compare_net()
        return ret

    def check_arguments_valid(self: any, args: argparse.Namespace) -> None:
        """
        Check arguments valid, if invalid, throw exception
        """
        check_pass = True
        if len(args.fusion_rule_file) != 0 \
                or len(args.quant_fusion_rule_file) != 0 \
                or len(args.custom_script_path) != 0:
            check_pass = False

        if args.op_name is not None \
                or args.dump_version != ConstManager.BINARY_DUMP_TYPE \
                or args.mapping:
            check_pass = False

        if not check_pass:
            log.print_error_log('The argument [-f|-q|-c|-op|-map|-v] is not supported'
                                ' in pytorch precision comparison scenarios.')
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

        path_type = path_check.PathType.File
        ret = path_check.check_output_path_valid(self.output_path, False, path_type)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)

    def _write_header_to_file(self: any) -> bool:
        """
        Write the result file header into the file.
        """
        try:
            with os.fdopen(os.open(self.output_path, ConstManager.WRITE_FLAGS,
                                   ConstManager.WRITE_MODES), 'a+', newline='') as output_file:
                compare_result_header = compare_result.get_result_title(self.algorithm_manager, self.OP_HEADER)
                writer = csv.writer(output_file)
                if writer is not None:
                    sanitized_compare_result_header = [sanitize_csv_value(cell) for cell in compare_result_header]
                    writer.writerow(sanitized_compare_result_header)
            return True
        except IOError as io_error:
            log.print_open_file_error(self.output_path, io_error)
            return False
        finally:
            pass

    def _save_cmp_result_exec(self: any, result: list) -> None:
        with os.fdopen(os.open(self.output_path, ConstManager.WRITE_FLAGS,
                               ConstManager.WRITE_MODES), 'a+', newline='') as output_file:
            writer = csv.writer(output_file)
            for item in result:
                sanitized_row = [sanitize_csv_value(cell) for cell in item]
                writer.writerow(sanitized_row)

    def _save_cmp_result(self: any, result: list, lock: any = None) -> None:
        """
        Write the compare result to result file.
        """
        if lock is not None:
            lock.acquire()
        try:
            self._save_cmp_result_exec(result)
        except IOError as io_error:
            log.print_open_file_error(self.output_path, io_error)
        finally:
            if lock is not None:
                lock.release()

    def _get_compare_dump_data(self: any, op_name: str, my_dump_dataset: str, golden_dataset: str) -> any:
        """
        Get compare data and check it
        """
        my_dump_data, golden_dump_data, _ = self.compare_data.get_dump_data(
            my_dump_dataset, golden_dataset)
        if my_dump_data.size != golden_dump_data.size:
            message = log.print_cannot_compare_warning(
                op_name, '(%d)' % my_dump_data.size, '(%d)' % golden_dump_data.size)
            raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR, message)
        return my_dump_data.flatten(), golden_dump_data.flatten(), my_dump_data.shape

    def _do_compare_tensor(self: any, op_name: str, my_dump_dataset: str, golden_dataset: str) -> any:
        my_dump_data, golden_dump_data, shape = self._get_compare_dump_data(
            op_name, my_dump_dataset, golden_dataset)
        my_dump_data_type = str(my_dump_data.dtype)
        algorithm_result, fail_reason = self.algorithm_manager.compare(
            my_dump_data, golden_dump_data,
            {'my_output_dump_file': my_dump_dataset, 'ground_truth_dump_file': golden_dataset,
             'shape_type': utils.get_shape_type(shape)})
        return [my_dump_data_type, shape], algorithm_result, fail_reason

    def _compare_tensor(self: any, order: int, op_name: str, ext_opname: str, my_dump_dataset: str) -> any:
        """
        compare one tensor
        :order: order
        :name: name
        :ext_opname: extend_opname
        :my_dump_dataset: dataset path for compare
        """
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager)
        match, golden_dataset, fail_reason = self.compare_data.get_golden_dataset(ext_opname, my_dump_dataset)

        if not match:
            return match, fusion_op_result.get_pytorch_result(
                compare_result.PytorchOpInfo(order, op_name, my_dump_dataset, ConstManager.NAN), None, [fail_reason])
        op_info = compare_result.PytorchOpInfo(order, op_name, my_dump_dataset, golden_dataset)
        try:
            return match, self._do_compare_and_get_result([my_dump_dataset, golden_dataset],
                                                          fusion_op_result, op_info)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                CompareError) as err:
            error_msg = [err.message] if isinstance(err, CompareError) else [str(err)]
            return match, fusion_op_result.get_pytorch_result(op_info, None, error_msg)
        finally:
            pass

    def _do_compare_and_get_result(self: any, dataset_list: list,
                                   fusion_op_result: compare_result.FusionOpComResult,
                                   op_info: compare_result.PytorchOpInfo) -> list:
        type_shape_list, algorithm_result, fail_reason = self._do_compare_tensor(op_info.op_name, dataset_list[0],
                                                                                 dataset_list[1])
        tensor_info = {
            "tensor_id": None,
            "shape": type_shape_list[1],
            "my_output_dtype": type_shape_list[0]
        }
        return fusion_op_result.get_pytorch_result(
                op_info,
                [compare_result.TensorResult(tensor_info, [algorithm_result, ''], fail_reason, False)],
                fail_reason)

    def _compare_one_op(self: any, order: int, ext_opname: str, lock: any) -> None:
        """
        Compare the dump data corresponding to ext_opname.
        """
        my_dump_datasets = self.compare_data.get_my_dump_datasets(ext_opname)
        op_name = self.compare_data.get_original_opname(ext_opname)
        log.print_info_log('[{0}] Start to compare "/{0}/{1}".'.format(op_name, order))
        if not my_dump_datasets:
            log.print_warn_log("[{}] has no comparable data.".format(op_name))
            return

        not_match = []
        op_cmp_result = []
        for my_dump_dataset in my_dump_datasets:
            match, tensor_result = self._compare_tensor(order, op_name, ext_opname, my_dump_dataset)
            not_match.append(not match)
            op_cmp_result += tensor_result
        if all(not_match):
            op_cmp_result.clear()
            fail_reason = 'No data match with op:{} in golden dump data.'.format(op_name)
            op_cmp_result = compare_result.FusionOpComResult(self.algorithm_manager).get_pytorch_result(
                compare_result.PytorchOpInfo(order, op_name, ConstManager.NAN, ConstManager.NAN), None, [fail_reason])
        self._save_cmp_result(op_cmp_result, lock)

    def _compare_in_one_process(self: any, order_group: list, lock: any) -> None:
        """
        Compare all the ops in the order_group.
        """
        # the parallel reading of the h5py file needs to be \
        # opened separately in the process.
        self.compare_data.open_file('r')
        try:
            for order in order_group:
                for ext_opname in self.compare_data.get_ext_opname_by_order(order):
                    self._compare_one_op(order, ext_opname, lock)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                CompareError) as ex:
            log.print_warn_log("Failed to compare with except: " + str(ex))
        finally:
            self.compare_data.close_file()

    def _compare_by_multi_process(self: any) -> None:
        """
        Compare for all the pytorch op by multi process
        """
        process_num = int((multiprocessing.cpu_count() + 1) / 2)
        all_orders = self.compare_data.get_all_orders()
        orders_groups = (all_orders[i::process_num] for i in range(process_num))

        # start multiple processes for parallel comparison.
        pool = multiprocessing.Pool(process_num)
        lock = multiprocessing.Manager().RLock()
        for order_group in orders_groups:
            pool.apply_async(self._compare_in_one_process, args=(order_group, lock))

        pool.close()
        pool.join()

    def _filter_one_line(self: any, result_path: str, row: list, csv_writer: any, position: list) -> None:
        cos_index = position[0]
        my_dump_index = position[1]
        golden_index = position[2]
        if row[cos_index] == ConstManager.NAN or float(row[cos_index]) <= 0.95:
            sanitized_row = [sanitize_csv_value(cell) for cell in row]
            csv_writer.writerow(sanitized_row)
            if row[my_dump_index] != ConstManager.NAN:
                tensor_data = self.compare_data.my_dump.get_dump_data(row[my_dump_index])
                dump_data_path = os.path.join(result_path, "my_dump" + row[my_dump_index])
                self._save_numpy_data(dump_data_path, tensor_data)

            if row[golden_index] != ConstManager.NAN:
                tensor_data = self.compare_data.golden_dump.get_dump_data(row[golden_index])
                dump_data_path = os.path.join(result_path, "gold_dump" + row[golden_index])
                self._save_numpy_data(dump_data_path, tensor_data)

    def _filter_result_process(self: any,
                               result_file_handle: any,
                               filtered_file_handle: any,
                               filtered_result_path: str) -> None:
        csv_reader = csv.reader(result_file_handle)
        csv_writer = csv.writer(filtered_file_handle)
        line_num = 0
        location = []
        for row in csv_reader:
            if line_num == 0:
                location = self._get_item_location(row)
                sanitized_row = [sanitize_csv_value(cell) for cell in row]
                csv_writer.writerow(sanitized_row)
            else:
                self._filter_one_line(filtered_result_path, row, csv_writer, location)
            line_num = line_num + 1

    def _filter_compare_result(self: any) -> None:
        result_filename_base, _ = os.path.splitext(os.path.basename(self.output_path))
        filtered_result_folder_name = "{}_{}".format(result_filename_base, "filtered")
        filtered_result_folder_path = os.path.join(os.path.dirname(self.output_path), filtered_result_folder_name)
        if not os.path.exists(filtered_result_folder_path):
            os.makedirs(filtered_result_folder_path, mode=0o700)

        filtered_result_file = os.path.join(filtered_result_folder_path, "filtered_result.csv")
        self.compare_data.open_file('r')
        try:
            with open(self.output_path, 'r') as result_file:
                with os.fdopen(os.open(filtered_result_file, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                               'w+') as filtered_file:
                    self._filter_result_process(result_file, filtered_file, filtered_result_folder_path)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            log.print_error_log('Failed to filtering pytorch compare result!, {}'.format(str(error)))
        finally:
            self.compare_data.close_file()

    def _compare_net(self: any) -> int:
        """
        compare vector for all fusion ops, and write result to file.
        """
        log.print_info_log("pytorch precision compare start!")
        self._compare_by_multi_process()
        self.record_not_matched()
        utils.sort_result_file_by_index(self.output_path)

        if os.path.exists(self.output_path):
            log.print_write_result_info('comparison result', self.output_path)
            if self.filter_flag:
                self._filter_compare_result()
            if self.is_open_advisor:
                self._do_advisor()
            return CompareError.MSACCUCMP_NONE_ERROR
        return CompareError.MSACCUCMP_UNKNOWN_ERROR

    def _do_advisor(self):
        try:
            from advisor.compare_advisor import CompareAdvisor
        except ImportError as import_error:
            log.print_warn_log("Unable to import module: %s." % str(import_error))
            log.print_warn_log("Skip compare results Analysis.")
        else:
            out_path = os.path.dirname(self.output_path)
            compare_advisor = CompareAdvisor(self.output_path, [], out_path)
            advisor_result = compare_advisor.advisor()
            message_list = advisor_result.print_advisor_log()
            advisor_result.gen_summary_file(out_path, message_list)

