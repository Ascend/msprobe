#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from collections import namedtuple
import re

from utils import transplant_logger as translog

MIN_ARGS_NUM = 0
CudaOp = namedtuple('CudaOp', ['file_path', 'func_name', 'min_args_num', 'max_args_num'])
CPP_FUNC_SUB_RE_PATTERN = re.compile('&|\(|\)')
TYPE_DECLARE_RE_PATTERN = re.compile('<.*?>')
FUNC_NAMES_RE_PATTERN = re.compile('"(.*?)"')
INIT_FUNC_RE_PATTERN = re.compile('::init<.*?>')
LAMBDA_ARG_RE_PATTERN = re.compile('\[\]\((.*?)\)')
TORCH_FN_RE_PATTERN = re.compile('TORCH_FN\((.*?)\)')
MAX_STRING_LENGTH = 10000


def re_limit_length(pattern, string, func):
    if len(string) > MAX_STRING_LENGTH:
        translog.warning(f'The character string contains more than {MAX_STRING_LENGTH} characters, '
                         f'regular expression matching is skipped.')
        return func(pattern, '')
    else:
        return func(pattern, string)


class _DeclareLineParser:
    def __init__(self, cuda_ops_list, file_lines, rel_file_path):
        self.cuda_ops = cuda_ops_list
        self._file_lines = file_lines
        self.rel_file_path = rel_file_path

    def parse_class_declare(self, func_line):
        # deal with m.class_<GPUDecoder>("GPUDecoder").def(torch::init<std::string, torch::Device>())
        #  "----.def("next", &GPUDecoder::decode);
        names = re_limit_length(FUNC_NAMES_RE_PATTERN, func_line, re.findall)
        if not names:
            return
        class_name = names[0]
        class_init_func = re_limit_length(INIT_FUNC_RE_PATTERN, func_line, re.search)
        if not class_init_func:
            self.cuda_ops.append(CudaOp(self.rel_file_path, class_name, MIN_ARGS_NUM, -1))
        else:
            class_init_func = class_init_func.group()
            if '<>' in class_init_func:
                args_name = 0
            else:
                args_name = class_init_func.count(',') + 1
            self.cuda_ops.append(CudaOp(self.rel_file_path, class_name, args_name, args_name))
        if len(names) <= 1:
            return
        for name in names[1:]:
            # instance api ignore args num
            func_name = f'{names[0]}.{name}'.replace('::', '.')
            self.cuda_ops.append(CudaOp(self.rel_file_path, func_name, MIN_ARGS_NUM, -1))

    def _parse_cpp_func_args_num(self, cpp_func_name):
        # deal with m.def("get_indice_pairs_2d", &spconv::getIndicePair<2>, "get_indice_pairs_2d");
        cpp_func_name = re.sub(CPP_FUNC_SUB_RE_PATTERN, '', cpp_func_name).split('<')[0]
        in_func_def_line = False
        func_def_line = ''
        for row, line in enumerate(self._file_lines):
            # escape annotation
            if line.strip().startswith('/'):
                continue
            line = line.split('//')[0].strip()
            # def with c10::Dict<std::string, c10::Dict<std::string, double>> GPUDecoder::
            # "----get_metadata() const {xxx}
            if line.endswith('::') and row + 1 < len(self._file_lines):
                line += self._file_lines[row + 1].strip()
            if f' {cpp_func_name}(' in line:
                in_func_def_line = True
            if in_func_def_line:
                if "{" in line:
                    func_def_line += line.split('{')[0]
                    break
                else:
                    func_def_line += line
        if not func_def_line:
            return MIN_ARGS_NUM, -1
        if f'{cpp_func_name}()' in func_def_line:
            return MIN_ARGS_NUM, MIN_ARGS_NUM
        # escape changeable params
        if '...' in func_def_line:
            return MIN_ARGS_NUM, -1
        # escape at::TensorAccessor<scalar_t, 2> waveform_accessor
        type_sep_count = sum(
            type_sep.count(',') for type_sep in re_limit_length(TYPE_DECLARE_RE_PATTERN, func_def_line, re.findall))
        sep_count = func_def_line.count(',') - type_sep_count
        return sep_count + 1, sep_count + 1


class PybindModuleParser(_DeclareLineParser):
    def __init__(self, cuda_ops_list, file_lines, rel_file_path):
        super().__init__(cuda_ops_list, file_lines, rel_file_path)

    def parse_m_def(self, m_def_line):
        names = re_limit_length(FUNC_NAMES_RE_PATTERN, m_def_line, re.findall)
        if not names:
            return
        func_name = names[0].replace('::', '.')
        if 'py::arg' in m_def_line:
            # deal with m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)", py::arg("input");
            elements = m_def_line.split(',')
            min_args_num = 0
            max_args_num = 0
            for element in elements:
                if not element.strip().startswith('py::arg'):
                    continue
                max_args_num += 1
                if '=' not in element:
                    min_args_num += 1
        else:
            # deal with m.def("forward", &chamfer_forward, "chamfer forward (CUDA)");
            if len(m_def_line.split(',')) <= 1:
                min_args_num, max_args_num = MIN_ARGS_NUM, -1
            else:
                cpp_func_name = m_def_line.split(',')[1].split(')')[0].strip()
                min_args_num, max_args_num = self._parse_cpp_func_args_num(cpp_func_name)
        self.cuda_ops.append(CudaOp(self.rel_file_path, func_name, min_args_num, max_args_num))


class TorchLibraryParser(_DeclareLineParser):
    def __init__(self, cuda_ops_list, file_lines, rel_file_path):
        super().__init__(cuda_ops_list, file_lines, rel_file_path)

    def parse_m_def(self, func_line):
        double_quotes = '::'
        if len(func_line.split('"')) <= 1:
            return
        func_name = func_line.split('"')[1].replace(double_quotes, '.')
        if '(' in func_name:
            # deal with m.def(TORCH_SELECTIVE_SCHEMA(
            #  "----"torchvision::roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height
            #  "----int pooled_width) -> (Tensor, Tensor)"));
            new_func_name = func_name.split('(')[0]
            if '()' in func_name:
                min_args_num = max_args_num = MIN_ARGS_NUM
            else:
                min_args_num = max_args_num = func_name.split(')')[0].count(',') + 1
            func_name = new_func_name
        elif '[](' in func_line:
            # def with m.def("torchaudio::ffmpeg_set_log_level", [](int64_t level) {
            # "----av_log_set_level(static_cast<int>(level));
            #   });
            func_name = func_line.split('"')[1].replace(double_quotes, '.')
            arg_declare = re_limit_length(LAMBDA_ARG_RE_PATTERN, func_line, re.findall)
            if not arg_declare:
                return
            arg_declare = arg_declare[0]
            if not arg_declare:
                min_args_num = max_args_num = MIN_ARGS_NUM
            else:
                min_args_num = max_args_num = arg_declare.count(',') + 1
        else:
            # deal with m.def("_cuda_version", &cuda_version);
            # deal with m.def("read_video_from_file", read_video_from_file);
            func_name = func_line.split('"')[1].replace(double_quotes, '.')
            if len(func_line.split(',')) <= 1:
                min_args_num, max_args_num = MIN_ARGS_NUM, -1
            else:
                cpp_func_name = func_line.split(',')[1].split(')')[0].strip()
                min_args_num, max_args_num = self._parse_cpp_func_args_num(cpp_func_name)
        self.cuda_ops.append(CudaOp(self.rel_file_path, func_name, min_args_num, max_args_num))

    def parse_m_impl(self, func_line):
        if 'TORCH_SELECTIVE_NAME' in func_line and 'TORCH_FN' in func_line:
            # deal with m.impl(
            #   "----TORCH_SELECTIVE_NAME("torchvision::roi_align"),
            #   "----TORCH_FN(qroi_align_forward_kernel));
            if len(func_line.split('"')) <= 1:
                return
            func_name = func_line.split('"')[1].replace('::', '.')
            cpp_func_name = re_limit_length(TORCH_FN_RE_PATTERN, func_line, re.findall)
            if not cpp_func_name:
                min_args_num, max_args_num = MIN_ARGS_NUM, -1
            else:
                cpp_func_name = cpp_func_name[0]
                min_args_num, max_args_num = self._parse_cpp_func_args_num(cpp_func_name)
        else:
            # deal with m.impl("rnnt_loss", &compute);
            names = re_limit_length(FUNC_NAMES_RE_PATTERN, func_line, re.findall)
            func_name = names[0].replace('::', '.')
            if len(func_line.split(',')) <= 1:
                min_args_num, max_args_num = MIN_ARGS_NUM, -1
            else:
                cpp_func_name = func_line.split(',')[1].split(')')[0].strip()
                min_args_num, max_args_num = self._parse_cpp_func_args_num(cpp_func_name)
        self.cuda_ops.append(CudaOp(self.rel_file_path, func_name, min_args_num, max_args_num))
