#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import traceback
import atexit
import logging
import os
import stat
import shutil
import pandas as pd
import torch


class Logger(logging.Logger):

    def __init__(self, log_level=logging.INFO,
                 log_format='%(asctime)s [%(levelname)s] %(message)s',
                 datefmt='%Y-%m-%d %H:%M:%S'):
        super().__init__('', log_level)
        self._formatter = logging.Formatter(log_format, datefmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._formatter)
        self.addHandler(console_handler)


translog = Logger()


class ShapeRange:
    def __init__(self):
        self.min_shape_dict = {}  # key: shape_len, value: shape
        self.max_shape_dict = {}
        self.max_shape_len = 0

    def __str__(self):
        shape_range_string_list = []
        for i in range(1, self.max_shape_len + 1):
            if i not in self.max_shape_dict or i not in self.min_shape_dict:
                continue
            shape_range_string_list.append(f"{self.min_shape_dict.get(i)}-{self.max_shape_dict.get(i)}")
        return "[" + " | ".join(shape_range_string_list) + "]" if shape_range_string_list else "[()]"

    def update(self, shape):
        length = len(shape)
        shape = tuple(list(int(_shape) for _shape in shape))
        if length not in self.max_shape_dict:
            self.max_shape_dict[length] = shape
        if length not in self.min_shape_dict:
            self.min_shape_dict[length] = shape
        new_max_shape_list = []
        new_min_shape_list = []
        for shape_value, max_shape, min_shape in zip(
                shape, self.max_shape_dict.get(length), self.min_shape_dict.get(length)):
            new_max_shape_list.append(max(shape_value, max_shape))
            new_min_shape_list.append(min(shape_value, min_shape))
        self.max_shape_dict[length] = tuple(new_max_shape_list)
        self.min_shape_dict[length] = tuple(new_min_shape_list)
        self.max_shape_len = max(self.max_shape_len, length)


class TraceInfo:
    def __init__(self, stack, api_name, call_number):
        self.input_shape_list = None
        self.output_shape_list = None
        self.api_name = api_name
        self.stack = stack
        self.call_number = call_number
        self.input_shape_range = []
        self.output_shape_range = []

    @staticmethod
    def _update_shape_range(shape_list, range_list):
        diff_len = len(shape_list) - len(range_list)
        for _ in range(diff_len):
            range_list.append(ShapeRange())

        for shape_range, shape in zip(range_list, shape_list):
            shape_range.update(shape)

    def update_input_shape_range(self, input_shape_list):
        self._update_shape_range(input_shape_list, self.input_shape_range)

    def update_output_shape_range(self, output_shape_list):
        self._update_shape_range(output_shape_list, self.output_shape_range)


class DynamicShapeDetect:
    def __init__(self):
        self.trace_dict = {}
        self.count_dict = {}
        self.unique_trace_dict = {}
        self.dynamic_api_set = set()
        self.except_func = (
            'cpu', 'cuda', 'long', 'float', 'int', 'detach', 'to', 'contiguous',
            'type', 'requires_grad_', 'clone', 'numpy'
        )
        self.except_module = ('torch.nn.init.',)
        self.enable_hook = False
        self.start_hook = False
        self.__max_trace_dict_len = 10000

    def hook_func(self, func, func_name, call_number, *args, **kwargs):
        if not self._check_if_need_hook(func, func_name):
            return func(*args, **kwargs)
        trace = traceback.extract_stack()
        key = self._get_trace_info_key(func_name, trace, call_number)
        self._before_call(key, func, args, kwargs)
        if func_name == 'torch.jit.trace' and self.start_hook:
            self.enable_hook = False
        result = func(*args, **kwargs)
        if func_name == 'torch.jit.trace' and self.start_hook:
            self.enable_hook = True
        self._after_call(key, result)
        return result

    def start(self, dataset):
        self.start_hook = True
        self.enable_hook = True
        for data in iter(dataset):
            yield data
            for key in self.count_dict:
                self.count_dict[key] = -1

    def save(self):
        content = []
        for trace_info in self.dynamic_api_set:
            stack = []
            for line in trace_info.stack:
                if 'msft_dynamic_analysis' in line.filename:
                    continue
                line.lineno -= 1
                stack.append(line)
            if not stack:
                continue
            file_path = os.path.abspath(stack[-1].filename).replace(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".")
            str_stack = list(('<' + str(line)[len('<FrameSummary '):]) for line in stack)
            input_shape_range_str_list = list(str(shape_range) for shape_range in trace_info.input_shape_range)
            input_shape_range = ", ".join(input_shape_range_str_list) if trace_info.input_shape_range else "None"
            output_shape_range_str_list = list(str(shape_range) for shape_range in trace_info.output_shape_range)
            output_shape_range = ", ".join(output_shape_range_str_list) if trace_info.output_shape_range else "None"
            content.append((trace_info.api_name, file_path, stack[-1].lineno, str_stack, input_shape_range,
                            output_shape_range))
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file = os.path.join(script_dir, 'msft_dynamic_shape_analysis_report.csv')
        header = ('Function Name', 'File Path', 'Line Number', 'Call Stack', 'Input Shape Range', 'Output Shape Range')
        if os.path.exists(csv_file):
            translog.warning(f"The file {csv_file} already exists, it will be removed.")
            remove_path(csv_file)
        try:
            with os.fdopen(os.open(csv_file, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP),
                           'w+') as fp:
                data_frame = pd.DataFrame(columns=header)
                data_frame.to_csv(fp, index=False)
                new_data = pd.DataFrame(content)
                new_data.to_csv(fp, mode='a+', header=False, index=False)
        except Exception as e:
            raise RuntimeError("Can't open file: " + csv_file) from e
        if len(content) > 0:
            translog.warning(
                'It is detected that the model contains dynamic shapes, and it is recommended to enable binary when '
                'running the model!')

    def _after_call(self, key, result):
        output_shape_list = []
        if isinstance(result, torch.Tensor):
            output_shape_list.append(result.shape)
        elif isinstance(result, tuple):
            output_shape_list.extend([output.shape for output in result if isinstance(output, torch.Tensor)])
        unique_trace_info = self.unique_trace_dict.get(key[:key.rindex('_')])
        trace_info = self.trace_dict.get(key)
        unique_trace_info.update_output_shape_range(output_shape_list)
        if unique_trace_info not in self.dynamic_api_set:
            if trace_info.output_shape_list is None:
                trace_info.output_shape_list = output_shape_list
            elif trace_info.output_shape_list != output_shape_list:
                self.dynamic_api_set.add(unique_trace_info)

    def _before_call(self, key, func, args, kwargs):
        if hasattr(func, '__self__') and isinstance(func.__self__, torch.Tensor):
            args = [func.__self__] + list(args)
        input_shape_list = [arg.shape for arg in args if isinstance(arg, torch.Tensor)]
        input_shape_list.extend([arg.shape for arg in kwargs.values() if isinstance(arg, torch.Tensor)])
        unique_trace_info = self.unique_trace_dict.get(key[:key.rindex('_')])
        trace_info = self.trace_dict.get(key)
        unique_trace_info.update_input_shape_range(input_shape_list)
        if unique_trace_info not in self.dynamic_api_set:
            if trace_info.input_shape_list is None:
                trace_info.input_shape_list = input_shape_list
            elif trace_info.input_shape_list != input_shape_list:
                self.dynamic_api_set.add(unique_trace_info)

    def _get_trace_info_key(self, func_name, trace, call_number):
        key = f"{trace}_{func_name}_{call_number}"
        if key not in self.count_dict:
            self.count_dict[key] = -1
        if key not in self.unique_trace_dict:
            trace_info = TraceInfo(trace, func_name, call_number)
            self.unique_trace_dict[key] = trace_info
        self.count_dict[key] += 1
        key = f"{key}_{self.count_dict[key]}"
        if len(self.trace_dict) > self.__max_trace_dict_len:
            translog.warning("Excessive dynamic shape operator call stacks are detected. The program will exit.")
            exit(0)
        if key not in self.trace_dict:
            trace_info = TraceInfo(trace, func_name, call_number)
            self.trace_dict[key] = trace_info
        return key

    def _check_if_need_hook(self, func, func_name):
        if not self.enable_hook:
            return False
        if func_name.split('.')[-1] in self.except_func:
            return False
        if func_name.startswith(self.except_module):
            return False
        if 'torch' in func_name:
            return True
        if hasattr(func, '__self__') and isinstance(func.__self__, torch.Tensor):
            return True
        return False


@atexit.register
def save_report():
    try:
        DETECTOR.save()
    except PermissionError as exception:
        translog.error(f"Saving dynamic shape analysis report failed, the error message is: {exception}")
        return
    translog.info("Saving dynamic shape analysis report succeeded.")


def remove_path(path):
    try:
        if os.path.islink(os.path.abspath(path)) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except PermissionError as exp:
        raise Exception(f'Failed to delete {path}: {exp}') from exp


DETECTOR = DynamicShapeDetect()
