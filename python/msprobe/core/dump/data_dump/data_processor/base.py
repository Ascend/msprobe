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

import copy
import linecache
import os
import sys
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Tuple, Dict, Optional, Any, Union

import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import save_npy
from msprobe.core.common.log import logger
from msprobe.core.common.utils import convert_tuple, CompareException, is_np2


@dataclass
class ModuleForwardInputsOutputs:
    args: Optional[Tuple]
    kwargs: Optional[Dict]
    output: Any

    @property
    def args_tuple(self):
        return convert_tuple(self.args)

    @property
    def output_tuple(self):
        return convert_tuple(self.output)

    def update_output_with_args_and_kwargs(self):
        self.output = self.args + tuple(self.kwargs.values())


@dataclass
class ModuleBackwardInputsOutputs:
    grad_output: Optional[Tuple]
    grad_input: Optional[Tuple]

    @property
    def grad_input_tuple(self):
        return convert_tuple(self.grad_input)

    @property
    def grad_output_tuple(self):
        return convert_tuple(self.grad_output)


@dataclass
class ModuleBackwardInputs:
    grad_input: Optional[Tuple]

    @property
    def grad_input_tuple(self):
        return convert_tuple(self.grad_input)


@dataclass
class ModuleBackwardOutputs:
    grad_output: Optional[Tuple]

    @property
    def grad_output_tuple(self):
        return convert_tuple(self.grad_output)


class TensorStatInfo:
    def __init__(self, max_val=None, min_val=None, mean_val=None, norm_val=None, check_sum=None):
        self.max = max_val
        self.min = min_val
        self.mean = mean_val
        self.norm = norm_val
        self.check_sum = check_sum



class BaseDataProcessor:
    builtin_type = (bool, int, float, str, slice, type(Ellipsis))
    np_type = (np.integer, np.floating, np.bool_, np.complexfloating, np.str_, np.byte,
               np.str_ if is_np2() else np.unicode_, np.ndarray)

    def __init__(self, config, data_writer):
        self.data_writer = data_writer
        self.config = config
        if self.data_writer is not None:
            self.data_writer.config = config
        self.api_info_struct = {}
        self.stack_info_struct = {}
        self.current_api_or_module_name = None
        self.current_ignore_rules = {Const.FORWARD: None, Const.BACKWARD: None}
        self.api_data_category = None
        self.current_iter = 0
        self._return_forward_new_output = False
        self._forward_new_output = None
        self.save_name = None
        self.ignore_rules = self._load_builtin_ignore_rules()
        if hasattr(config, "data_mode"):
            self.allowed_data_mode = self._get_allowed_data_mode(config.data_mode)

    @property
    def data_path(self):
        return self.data_writer.dump_tensor_data_dir

    @property
    def is_terminated(self):
        return False

    @staticmethod
    def transfer_type(data):
        dtype = str(type(data))
        if 'int' in dtype:
            return int(data)
        elif 'float' in dtype:
            return float(data)
        else:
            return data

    @staticmethod
    def set_value_into_nested_structure(data_structure, indexes, value):
        '''
        Args:
            data_structure: nested data structure
            indexes: List
            value: value to be set
        '''
        if not indexes:
            raise ValueError("set_value_into_nested_structure failed: "
                             "indexes need to be non empty when set value to nested data structure")
        current_level = data_structure
        for i, index in enumerate(indexes):
            valid_for_list = isinstance(current_level, list) and isinstance(index, int) and len(current_level) > index
            valid_for_dict = isinstance(current_level, dict) and index in current_level
            is_last = i == len(indexes) - 1
            if valid_for_dict or valid_for_list:
                if is_last:
                    try:
                        current_level[index] = value
                    except Exception as e:
                        raise IndexError("set_value_into_nested_structure failed: passed indexes wrong") from e
                else:
                    try:
                        current_level = current_level[index]
                    except Exception as e:
                        raise IndexError("set_value_into_nested_structure failed: passed indexes wrong") from e
            else:
                raise ValueError("set_value_into_nested_structure failed: "
                                 "invalid data_structure type or invalid index")

    @staticmethod
    def is_distributed_op(module):
        return getattr(module, "op_is_distributed", False)

    @staticmethod
    def analyze_api_call_stack(name):
        if name.startswith("Primitive"):
            skip_frames = 4
        else:
            skip_frames = 5
        stack_info = []
        try:
            frame = sys._getframe(0)
            index = 0
            while frame is not None:
                if index >= skip_frames:
                    filename = frame.f_code.co_filename
                    lineno = frame.f_lineno
                    funcname = frame.f_code.co_name
                    stack_info.append((filename, lineno, funcname))
                frame = frame.f_back
                index += 1

            stack_str = []
            for path, line, func in stack_info:
                source_line = linecache.getline(path, line).strip()
                if source_line:
                    if any(filter_path in path for filter_path in Const.STACK_FILTER_KEYWORDS) or Const.CALL_STACK_FLAG in path:
                        continue
                    stack_line = f"File {path}, line {str(line)}, in {func}, \n {source_line}"
                    stack_str.append(stack_line)
            result = tuple(stack_str)
            return result
        except Exception as e:
            logger.warning(f"The call stack of <{name}> failed to retrieve, {e}.")
            return (Const.WITHOUT_CALL_STACK,)

    @staticmethod
    def _analyze_builtin(arg):
        single_arg = {}
        if isinstance(arg, slice):
            # The slice parameter may be of the tensor, numpy or other types.
            # It needs to be converted to the Python value type before JSON serialization
            single_arg.update({"type": "slice"})
            values = []
            for value in [arg.start, arg.stop, arg.step]:
                if value is not None:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"The data type {type(value)} cannot be converted to int type.")
                        value = None
                values.append(value)
            single_arg.update({"value": values})
        else:
            single_arg.update({"type": type(arg).__name__})
            # When arg is Ellipsis(...) type, it needs to be converted to str("...") type
            single_arg.update({"value": arg if arg is not Ellipsis else "..."})
        return single_arg

    @staticmethod
    def _analyze_numpy(arg):
        return {"type": type(arg).__name__, "value": arg.item()}

    @staticmethod
    def _analyze_ndarray(ndarray, _):
        ndarray_json = {}
        ndarray_json.update({'type': 'numpy.ndarray'})
        ndarray_json.update({'dtype': str(ndarray.dtype)})
        ndarray_json.update({'shape': ndarray.shape})

        # 先初始化默认值
        stats = {
            "Max": None,
            "Min": None,
            "Mean": None,
            "Norm": None
        }

        try:
            # 只有非空时才尝试计算
            if ndarray.size > 0:
                stats = {
                    "Max": np.max(ndarray).item(),
                    "Min": np.min(ndarray).item(),
                    "Mean": np.mean(ndarray).item(),
                    "Norm": np.linalg.norm(ndarray).item()
                }
        except Exception as e:
            # 决定打印内容或切片
            logger.warning(f"Error analyzing ndarray stats: {e}")

        # 最后一次性更新
        ndarray_json.update(stats)

        return ndarray_json

    @staticmethod
    def _get_allowed_data_mode(data_mode):
        if Const.ALL in data_mode:
            allowed_data_mode = [Const.FORWARD, Const.BACKWARD, Const.INPUT, Const.OUTPUT]
        else:
            allowed_data_mode = list(set(data_mode))
            if Const.FORWARD not in allowed_data_mode and Const.BACKWARD not in allowed_data_mode:
                allowed_data_mode += [Const.FORWARD, Const.BACKWARD]
            if Const.INPUT not in allowed_data_mode and Const.OUTPUT not in allowed_data_mode:
                allowed_data_mode += [Const.INPUT, Const.OUTPUT]
        return allowed_data_mode

    def _load_builtin_ignore_rules(self):
        return {}

    @staticmethod
    def _normalize_ignore_rules(ignore_rules):
        if not isinstance(ignore_rules, dict):
            return {Const.FORWARD: {}, Const.BACKWARD: {}}
        normalized_rules = {Const.FORWARD: {}, Const.BACKWARD: {}}
        for stage_name, rules in ignore_rules.items():
            if stage_name not in normalized_rules or not isinstance(rules, dict):
                continue
            normalized_stage_rules = {}
            for prefix_name, prefix_rules in rules.items():
                if not isinstance(prefix_name, str) or not isinstance(prefix_rules, dict):
                    continue
                normalized_prefix_rules = {}
                for api_name, value in prefix_rules.items():
                    if not isinstance(api_name, str):
                        continue
                    normalized_prefix_rules[api_name] = BaseDataProcessor._normalize_ignore_rule_value(value)
                if normalized_prefix_rules:
                    normalized_stage_rules[prefix_name] = normalized_prefix_rules
            normalized_rules[stage_name] = normalized_stage_rules
        return normalized_rules

    @staticmethod
    def _normalize_ignore_rule_value(value):
        normalized_rule = {
            "ignore_input_all": False,
            "ignore_output_all": False,
            "ignore_input_indices": set(),
            "ignore_output_indices": set(),
            "ignore_input_keys": set(),
            "ignore_output_keys": set()
        }
        if not isinstance(value, dict):
            return normalized_rule

        for io_type, key_all, key_indices in (
            (Const.INPUT, "ignore_input_all", "ignore_input_indices"),
            (Const.OUTPUT, "ignore_output_all", "ignore_output_indices")
        ):
            io_value = value.get(io_type)
            if io_value == Const.ALL:
                normalized_rule[key_all] = True
                continue
            if not isinstance(io_value, list):
                continue
            normalized_rule[key_indices].update(index for index in io_value if isinstance(index, int) and index >= 0)
            normalized_rule[f"ignore_{io_type}_keys"].update(
                key for key in io_value if isinstance(key, str) and key
            )
        return normalized_rule

    @staticmethod
    def _parse_ignore_key(full_name):
        if not full_name:
            return None

        # API name format in dump:
        # 1) <Prefix>.<Api>.<CallId>.<Stage>  (main path)
        # 2) <Prefix>.<Api>.<Stage>          (fallback)
        parts = full_name.split(Const.SEP)
        if len(parts) < 2:
            return None

        if parts[-1] in (Const.FORWARD, Const.BACKWARD):
            if len(parts) >= 3 and parts[-2].isdigit():
                base_parts = parts[:-2]
            else:
                base_parts = parts[:-1]
        elif len(parts) >= 3 and parts[-2] in (Const.FORWARD, Const.BACKWARD) and parts[-1].isdigit():
            base_parts = parts[:-2]
        else:
            base_parts = parts

        if len(base_parts) < 2:
            return None

        prefix_name = base_parts[0]
        api_name = Const.SEP.join(base_parts[1:])
        return prefix_name, api_name

    def _resolve_ignore_rule(self, api_or_module_name, stage):
        parsed_key = self._parse_ignore_key(api_or_module_name)
        if not parsed_key:
            return None
        prefix_name, api_name = parsed_key
        return self.ignore_rules.get(stage, {}).get(prefix_name, {}).get(api_name)

    def _get_ignore_rule(self, stage):
        return self.current_ignore_rules.get(stage)

    def _apply_ignore_rules(self, data, stage, io_type):
        ignore_rule = self._get_ignore_rule(stage)
        if not ignore_rule:
            return data
        ignore_all = ignore_rule["ignore_input_all"] if io_type == Const.INPUT else ignore_rule["ignore_output_all"]
        ignore_indices = ignore_rule["ignore_input_indices"] if io_type == Const.INPUT \
            else ignore_rule["ignore_output_indices"]
        ignore_keys = ignore_rule["ignore_input_keys"] if io_type == Const.INPUT else ignore_rule["ignore_output_keys"]
        if ignore_all:
            if isinstance(data, tuple):
                return tuple(None for _ in data)
            if isinstance(data, list):
                return [None for _ in data]
            if isinstance(data, dict):
                return {key: None for key in data}
            return None

        hit_items = []
        if isinstance(data, (list, tuple)) and ignore_indices:
            masked_data = list(data)
            for index in ignore_indices:
                if 0 <= index < len(masked_data):
                    masked_data[index] = None
                    hit_items.append(str(index))
            if hit_items:
                logger.debug(f"Ignore rules matched for {self.current_api_or_module_name}: "
                             f"{stage}.{io_type}.{','.join(sorted(hit_items))}")
            return tuple(masked_data) if isinstance(data, tuple) else masked_data

        if isinstance(data, dict) and ignore_keys:
            masked_data = dict(data)
            for key in ignore_keys:
                if key in masked_data:
                    masked_data[key] = None
                    hit_items.append(str(key))
            if hit_items:
                logger.debug(f"Ignore rules matched for {self.current_api_or_module_name}: "
                             f"{stage}.{io_type}.{','.join(sorted(hit_items))}")
            return masked_data

        return data

    @classmethod
    def get_special_types(cls):
        return cls.builtin_type + cls.np_type

    @classmethod
    def recursive_apply_transform(cls, args, transform, depth=0, key_stack=None) -> Union[dict, list, None]:
        if key_stack is None:
            key_stack = []
        if depth > Const.DUMP_MAX_DEPTH:
            logger.error(f"The maximum depth of recursive transform, {Const.DUMP_MAX_DEPTH} is reached.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(args, cls.get_special_types()):
            arg_transform = transform(args, key_stack)
            return arg_transform
        elif isinstance(args, tuple) and hasattr(args, '_fields'):
            # namedtuple to dict
            args_dict = {field: getattr(args, field) for field in args._fields}
            return cls.apply_transform_dict(args_dict, transform, depth, key_stack)
        elif is_dataclass(args):
            # dataclass to dict
            args_dict = {field: getattr(args, field) for field in args.__dataclass_fields__}
            return cls.apply_transform_dict(args_dict, transform, depth, key_stack)
        elif isinstance(args, (list, tuple)):
            result_list = cls.apply_transform_list(args, transform, depth, key_stack)
            return result_list
        elif isinstance(args, dict):
            return cls.apply_transform_dict(args, transform, depth, key_stack)
        elif args is not None:
            logger.debug(f"Data type {type(args)} is not supported.")
            return None
        else:
            return None

    @classmethod
    def apply_transform_dict(cls, args, transform, depth, key_stack):
        result_dict = {}
        for k, arg in args.items():
            key_stack.append(k)
            try:
                result_dict[k] = cls.recursive_apply_transform(
                    arg, transform, depth=depth + 1, key_stack=key_stack
                )
            finally:
                key_stack.pop()
        return result_dict

    @classmethod
    def apply_transform_list(cls, args, transform, depth, key_stack):
        result_list = []
        for i, arg in enumerate(args):
            key_stack.append(i)
            try:
                result_list.append(cls.recursive_apply_transform(
                    arg, transform, depth=depth + 1, key_stack=key_stack
                ))
            finally:
                key_stack.pop()
        return result_list

    @classmethod
    def register_hook_single_element(cls, element, suffix_stack, hook_fn):
        if cls.is_hookable_element(element):
            indexes = copy.deepcopy(suffix_stack)
            wrap_hook_fn = partial(hook_fn, indexes=indexes)

            def real_hook_fn(grad):
                return wrap_hook_fn(grad)

            element.register_hook(real_hook_fn)

    def if_return_forward_new_output(self):
        return self._return_forward_new_output

    def get_forward_new_output(self):
        self._return_forward_new_output = False
        return self._forward_new_output

    def update_iter(self, current_iter):
        self.current_iter = current_iter

    def update_api_or_module_name(self, api_or_module_name):
        if self.current_api_or_module_name != api_or_module_name:
            self.current_api_or_module_name = api_or_module_name
            self.current_ignore_rules = {
                Const.FORWARD: self._resolve_ignore_rule(api_or_module_name, Const.FORWARD),
                Const.BACKWARD: self._resolve_ignore_rule(api_or_module_name, Const.BACKWARD)
            }

    def is_dump_for_data_mode(self, forward_backward, input_output):
        """
        Compare the parameters with data_mode to determine whether to dump.

        Args:
            forward_backward(str): The forward or backward mode to check.
            input_output(str): The input or output mode to check.

        Return:
            bool: True if the parameters are in data_mode or data_mode is all, False otherwise.
        """
        return forward_backward in self.allowed_data_mode and input_output in self.allowed_data_mode

    def analyze_element(self, element):
        return self.recursive_apply_transform(element, self.analyze_single_element)

    def analyze_forward_input(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = {}
        # check whether data_mode contains forward or input
        if self.is_dump_for_data_mode(Const.FORWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            args_info_list = self.analyze_element(
                self._apply_ignore_rules(module_input_output.args_tuple, Const.FORWARD, Const.INPUT)
            )
            api_info_struct[name][Const.INPUT_ARGS] = args_info_list
            self.api_data_category = Const.KWARGS
            kwargs_info_list = self.analyze_element(
                self._apply_ignore_rules(module_input_output.kwargs, Const.FORWARD, Const.INPUT)
            )
            api_info_struct[name][Const.INPUT_KWARGS] = kwargs_info_list

        return api_info_struct

    def analyze_forward_output(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        if self.is_distributed_op(module):
            module_input_output.update_output_with_args_and_kwargs()
        api_info_struct = {}
        # check whether data_mode contains forward or input
        if self.is_dump_for_data_mode(Const.FORWARD, Const.OUTPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(
                self._apply_ignore_rules(module_input_output.output_tuple, Const.FORWARD, Const.OUTPUT)
            )
            api_info_struct[name][Const.OUTPUT] = output_info_list

        return api_info_struct

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = {}
        # check whether data_mode contains forward or input
        if self.is_dump_for_data_mode(Const.FORWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            args_info_list = self.analyze_element(
                self._apply_ignore_rules(module_input_output.args_tuple, Const.FORWARD, Const.INPUT)
            )
            api_info_struct[name][Const.INPUT_ARGS] = args_info_list
            self.api_data_category = Const.KWARGS
            kwargs_info_list = self.analyze_element(
                self._apply_ignore_rules(module_input_output.kwargs, Const.FORWARD, Const.INPUT)
            )
            api_info_struct[name][Const.INPUT_KWARGS] = kwargs_info_list

        # check whether data_mode contains forward or output
        if self.is_dump_for_data_mode(Const.FORWARD, Const.OUTPUT):
            api_info_struct[name] = api_info_struct.get(name, {})
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(
                self._apply_ignore_rules(module_input_output.output_tuple, Const.FORWARD, Const.OUTPUT)
            )
            api_info_struct[name][Const.OUTPUT] = output_info_list

        if name in api_info_struct and hasattr(module_input_output, Const.PARAMS):
            self.api_data_category = Const.PARAMS
            api_info_struct[name][Const.PARAMS] = self.analyze_element(getattr(module_input_output, Const.PARAMS))

        return api_info_struct

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.BACKWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            input_info_list = self.analyze_element(module_input_output.grad_input_tuple)
            input_info_list = self._apply_ignore_rules(input_info_list, Const.BACKWARD, Const.INPUT)
            api_info_struct[name][Const.INPUT] = input_info_list

        if self.is_dump_for_data_mode(Const.BACKWARD, Const.OUTPUT):
            api_info_struct[name] = api_info_struct.get(name, {})
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(module_input_output.grad_output_tuple)
            output_info_list = self._apply_ignore_rules(output_info_list, Const.BACKWARD, Const.OUTPUT)
            api_info_struct[name][Const.OUTPUT] = output_info_list

        return api_info_struct

    def analyze_backward_input(self, name, module,
                               module_input_output: ModuleBackwardInputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.BACKWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT

            input_info_list = self.analyze_element(module_input_output.grad_input_tuple)
            input_info_list = self._apply_ignore_rules(input_info_list, Const.BACKWARD, Const.INPUT)
            api_info_struct[name][Const.INPUT] = input_info_list
        return api_info_struct

    def analyze_backward_output(self, name, module,
                                module_input_output: ModuleBackwardOutputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.BACKWARD, Const.OUTPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.OUTPUT

            output_info_list = self.analyze_element(module_input_output.grad_output_tuple)
            output_info_list = self._apply_ignore_rules(output_info_list, Const.BACKWARD, Const.OUTPUT)
            api_info_struct[name][Const.OUTPUT] = output_info_list
        return api_info_struct

    def analyze_params(self, name, param_name, grad):
        api_info_struct = {}
        self.save_name = name + Const.SEP + param_name
        data_info = self.analyze_element(grad)
        self.save_name = None
        grad_info_dict = {param_name: [data_info]}
        api_info_struct[name] = grad_info_dict
        return api_info_struct

    def get_save_file_path(self, suffix):
        file_format = Const.PT_SUFFIX if self.config.framework == Const.PT_FRAMEWORK else Const.NUMPY_SUFFIX
        if self.save_name is not None:
            dump_data_name = (self.save_name + file_format)
        else:
            suffix_with_seq = (Const.SEP + suffix) if suffix else ""
            dump_data_name = (self.current_api_or_module_name + Const.SEP + self.api_data_category + suffix_with_seq +
                              file_format)
        file_path = os.path.join(self.data_writer.dump_tensor_data_dir, dump_data_name)
        return dump_data_name, file_path

    def analyze_element_to_all_none(self, element):
        return self.recursive_apply_transform(element, lambda element, stack: None)

    def analyze_debug_forward(self, variable, name_with_count):
        self.current_api_or_module_name = name_with_count
        self.api_data_category = Const.DEBUG
        # these two attributes are used to construct tensor file name {name_with_count}.debug.{indexes}.npy/pt
        data_info = self.analyze_element(variable)
        return data_info

    def analyze_debug_backward(self, variable, grad_name_with_count_category, nested_data_structure):
        def hook_fn(grad, indexes):
            suffix = Const.SEP.join([str(index) for index in indexes])
            suffix_with_sep = (Const.SEP + suffix) if suffix else ""
            self.save_name = grad_name_with_count_category + suffix_with_sep
            grad_data_info = self.analyze_element(grad)
            self.save_name = None
            full_index = [grad_name_with_count_category] + indexes
            try:
                self.set_value_into_nested_structure(nested_data_structure, full_index, grad_data_info)
            except (ValueError, IndexError) as e:
                logger.warning(f"error occurred while recording statistics of {grad_name_with_count_category} variable,"
                               f"skip current recording, detailed information: {e}")
            return grad

        wrap_register_hook_single_element = partial(self.register_hook_single_element, hook_fn=hook_fn)
        self.recursive_apply_transform(variable, wrap_register_hook_single_element)

    def _analyze_and_save_ndarray(self, ndarray, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        save_npy(ndarray, file_path)
        ndarray_json = BaseDataProcessor._analyze_ndarray(ndarray, suffix)
        ndarray_json.update({"data_name": dump_data_name})
        return ndarray_json
