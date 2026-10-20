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

import inspect
from typing import Dict, Any, Optional, Callable, Union, List, Tuple

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml
from msprobe.core.common.log import logger


def _get_attr(module, attr_name):
    if Const.SEP in attr_name:
        sub_module_name, sub_attr = attr_name.rsplit(Const.SEP, 1)
        sub_module = getattr(module, sub_module_name, None)
        attr = getattr(sub_module, sub_attr, None)
    else:
        attr = getattr(module, attr_name, None)
    return attr


class ApiWrapper:
    def __init__(
        self,
        api_types: Dict[str, Dict[str, Any]],
        api_list_paths: Union[str, List[str], Tuple[str]],
        blacklist: Union[List[str], Tuple[str]] = None,
    ):
        self.api_types = api_types
        if not isinstance(api_list_paths, (list, tuple)):
            api_list_paths = [api_list_paths] * len(self.api_types)
        elif len(api_list_paths) != len(self.api_types):
            raise RuntimeError(
                "The number of api_list_paths must be equal to the number of frameworks in 'api_types', "
                "when api_list_paths is a list or tuple."
            )
        self.api_list_paths = api_list_paths
        self.blacklist = blacklist if blacklist else []
        self.api_names = self._get_api_names()
        self.wrapped_api_functions = dict()

    @staticmethod
    def deal_with_self_kwargs(api_name, api_func, args, kwargs):
        if kwargs and 'self' in kwargs:
            func_params = None
            try:
                func_params = inspect.signature(api_func).parameters
            except Exception:
                if api_name in Const.API_WITH_SELF_ARG:
                    func_params = inspect.signature(Const.API_WITH_SELF_ARG.get(api_name)).parameters
            if func_params is None:
                return False, args, kwargs

            for name, param in func_params.items():
                if name == 'self' and param.kind == inspect.Parameter.KEYWORD_ONLY:
                    return False, args, kwargs
            args_ = list(args)
            names_and_values = []
            self_index = 0
            for i, item in enumerate(func_params.items()):
                names_and_values.append((item[0], item[1].default))
                if item[0] == 'self':
                    self_index = i
                    break
            for i in range(len(args), self_index + 1):
                if names_and_values[i][0] in kwargs:
                    args_.append(kwargs.pop(names_and_values[i][0]))
                else:
                    args_.append(names_and_values[i][1])
            args = tuple(args_)

        return True, args, kwargs

    def wrap_api_func(self, api_name, api_func, prefix, hook_build_func, api_template):
        api_instance = api_template(api_name, api_func, prefix, hook_build_func)

        def api_function(*args, **kwargs):
            api_name_with_prefix = prefix + Const.SEP + str(api_name.split(Const.SEP)[-1])
            enable_wrap, args, kwargs = self.deal_with_self_kwargs(api_name_with_prefix, api_func, args, kwargs)
            if not enable_wrap:
                logger.warning(
                    f'Cannot collect precision data of {api_name_with_prefix}. '
                    'It may be fixed by passing the value of "self" '
                    'as a positional argument instead of a keyword argument. '
                )
                return api_func(*args, **kwargs)
            return api_instance(*args, **kwargs)

        for attr_name in Const.API_ATTR_LIST:
            if hasattr(api_func, attr_name):
                attr_value = getattr(api_func, attr_name)
                setattr(api_function, attr_name, attr_value)

        return api_function

    # pylint: disable=too-many-nested-blocks
    def wrap_api(self, api_templates, hook_build_func: Optional[Callable]):
        api_types_num = sum(len(v) for v in self.api_types.values())
        if not isinstance(api_templates, (list, tuple)):
            api_templates = [api_templates] * api_types_num
        elif len(api_templates) != api_types_num:
            raise RuntimeError(
                "The number of api_templates must be equal to the number of api_types, "
                "when api_templates is a list or tuple."
            )

        self.wrapped_api_functions.clear()
        index = 0
        for framework, api_types in self.api_types.items():
            wrapped_functions_in_framework = dict()
            for api_type, api_modules in api_types.items():
                wrapped_functions = dict()
                name_prefix = Const.API_DATA_PREFIX.get(framework, {}).get(api_type, "API")
                api_template = api_templates[index]
                index += 1
                for api_name in self.api_names.get(framework, {}).get(api_type, []):
                    ori_apis = []
                    api_source = {}
                    for module in api_modules[0]:
                        existed = False
                        for api in ori_apis:
                            if api == _get_attr(module, api_name):
                                existed = True
                                api_source[api].append(module)
                        if not existed:
                            api_source[_get_attr(module, api_name)] = [module]
                            ori_apis.append(_get_attr(module, api_name))
                    wrapped_functions[api_name] = []
                    for ori_api in ori_apis:
                        if callable(ori_api):
                            wrapped_api = self.wrap_api_func(
                                api_name, ori_api, name_prefix, hook_build_func, api_template
                            )
                            wrapped_functions[api_name].append((api_source.get(ori_api, []), wrapped_api))
                wrapped_functions_in_framework[api_type] = wrapped_functions
            self.wrapped_api_functions[framework] = wrapped_functions_in_framework
        return self.wrapped_api_functions

    def _get_api_names(self):
        api_names = dict()

        for index, framework in enumerate(self.api_types.keys()):
            api_list = load_yaml(self.api_list_paths[index]) or {}
            valid_names = dict()
            for api_type, api_modules in self.api_types.get(framework, {}).items():
                key_in_file = Const.SUPPORT_API_DICT_KEY_MAP.get(framework, {}).get(api_type)
                api_from_file = api_list.get(key_in_file) or []
                names = set()
                for api_name in api_from_file:
                    if f'{key_in_file}.{api_name}' in self.blacklist:
                        continue
                    target_attr = api_name
                    for module in api_modules[0]:
                        if Const.SEP in api_name:
                            sub_module_name, target_attr = api_name.rsplit(Const.SEP, 1)
                            target_module = getattr(module, sub_module_name, None)
                        else:
                            target_module = module
                        if target_module and target_attr in dir(target_module):
                            names.add(api_name)
                valid_names[api_type] = names
            api_names[framework] = valid_names

        return api_names


class ApiRegistry:
    """
    Base class for api registry.
    """

    def __init__(self, api_types, inner_used_api, supported_api_list_path, api_templates, blacklist=None):
        self.ori_api_attr = dict()
        self.wrapped_api_attr = dict()
        self.inner_used_ori_attr = dict()
        self.inner_used_wrapped_attr = dict()
        self.api_types = api_types
        self.inner_used_api = inner_used_api
        self.supported_api_list_path = supported_api_list_path
        self.api_templates = api_templates
        self.blacklist = blacklist if blacklist else []
        self.all_api_registered = False

    @staticmethod
    def store_ori_attr(ori_api_groups, api_list, api_ori_attr):
        for api in api_list:
            ori_api = []
            for ori_api_group in ori_api_groups:
                ori_api.append(([ori_api_group], _get_attr(ori_api_group, api)))
            api_ori_attr[api] = ori_api

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for api, api_attrs in attr_dict.items():
            for module, api_attr in api_attrs:
                if api_group in module:
                    if Const.SEP in api:
                        sub_module_name, sub_op = api.rsplit(Const.SEP, 1)
                        sub_module = getattr(api_group, sub_module_name, None)
                        if sub_module is not None:
                            setattr(sub_module, sub_op, api_attr)
                    else:
                        setattr(api_group, api, api_attr)

    @staticmethod
    def register_custom_api(module, api_name, api_prefix, hook_build_func, api_template):
        def wrap_api_func(api_name, api_func, prefix, hook_build_func, api_template):
            def api_function(*args, **kwargs):
                return api_template(api_name, api_func, prefix, hook_build_func)(*args, **kwargs)

            api_function.__name__ = api_name
            return api_function

        setattr(
            module,
            api_name,
            wrap_api_func(api_name, getattr(module, api_name), api_prefix, hook_build_func, api_template),
        )

    def register_all_api(self):
        self.all_api_registered = True
        for framework, api_types in self.api_types.items():
            for api_type, api_modules in api_types.items():
                api_type_with_framework = framework + Const.SEP + api_type
                for module in api_modules[1]:
                    self.set_api_attr(module, self.wrapped_api_attr.get(api_type_with_framework, {}))

    def register_inner_used_api(self):
        for api_type in self.inner_used_api.keys():
            self.set_api_attr(self.inner_used_api.get(api_type)[0], self.inner_used_wrapped_attr.get(api_type, {}))

    def restore_all_api(self):
        self.all_api_registered = False
        for framework, api_types in self.api_types.items():
            for api_type, api_modules in api_types.items():
                api_type_with_framework = framework + Const.SEP + api_type
                for module in api_modules[1]:
                    self.set_api_attr(module, self.ori_api_attr.get(api_type_with_framework, {}))

    def restore_inner_used_api(self):
        for api_type in self.inner_used_api.keys():
            self.set_api_attr(self.inner_used_api.get(api_type)[0], self.inner_used_ori_attr.get(api_type, {}))

    def initialize_hook(self, hook_build_func):
        api_wrapper = ApiWrapper(self.api_types, self.supported_api_list_path, self.blacklist)
        wrapped_api_functions = api_wrapper.wrap_api(self.api_templates, hook_build_func)

        for framework, api_types in self.api_types.items():
            for api_type, api_modules in api_types.items():
                ori_attr = dict()
                self.store_ori_attr(api_modules[0], api_wrapper.api_names.get(framework).get(api_type), ori_attr)
                api_type_with_framework = framework + Const.SEP + api_type
                self.ori_api_attr[api_type_with_framework] = ori_attr
                self.wrapped_api_attr[api_type_with_framework] = wrapped_api_functions.get(framework).get(api_type)

        for inner_used_api_type, inner_used_api_list in self.inner_used_api.items():
            ori_attr = dict()
            wrapped_attr = dict()
            for api_name in inner_used_api_list[1:]:
                if self.ori_api_attr.get(inner_used_api_type, {}).get(api_name):
                    ori_attr[api_name] = self.ori_api_attr.get(inner_used_api_type).get(api_name)
                    wrapped_attr[api_name] = self.wrapped_api_attr.get(inner_used_api_type).get(api_name)
            self.inner_used_ori_attr[inner_used_api_type] = ori_attr
            self.inner_used_wrapped_attr[inner_used_api_type] = wrapped_attr
