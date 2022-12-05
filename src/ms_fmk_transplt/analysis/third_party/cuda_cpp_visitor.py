from collections import namedtuple
import os
import re

import utils.trans_utils as utils

MIN_ARGS_NUM = 0
MAX_ARGS_NUM = 20
CudaOp = namedtuple('CudaOp', ['func_name', 'min_args_num', 'max_args_name'])


class CudaOpVisitor:
    def __init__(self, project_path):
        self.project_path = project_path
        self._cuda_ops = []
        self._file_lines = []

    @property
    def cuda_ops(self):
        return self._cuda_ops

    def visit_cuda_files(self):
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if not file.endswith('.cpp') and not file.endswith('.cu'):
                    continue
                file_path = os.path.join(root, file)
                if utils.islink(file_path):
                    continue
                utils.check_input_file_valid(file_path)
                self.visit_file(file_path)

    def visit_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file_reader:
            self._file_lines = file_reader.readlines()
        # parse cuda ops via "PYBIND11_MODULE"
        self._visit_py_bind_module()
        # parse cuda ops via "TORCH_LIBRARY"
        self._visit_torch_library_module()

    def _visit_py_bind_module(self):
        in_pybind_body = False
        in_func_or_class_declare = False
        declare_line = ''
        declare_lines = []
        for line in self._file_lines:
            # escape annotation
            if line.startswith('/'):
                continue
            line = line.split('//')[0].strip()
            if line.startswith('PYBIND11_MODULE('):
                in_pybind_body = True
            if not in_pybind_body:
                continue
            if line.startswith(('m.def(', 'py::class_')):
                if declare_line:
                    declare_lines.append(declare_line)
                    declare_line = ''
                in_func_or_class_declare = True
            if in_func_or_class_declare:
                declare_line += line
                if line.endswith('}'):
                    declare_lines.append(declare_line)
                    break

        for declare_line in declare_lines:
            if declare_line.startswith('m.def('):
                self._parse_bind_func(declare_line)
            elif declare_line.startswith('py::class_'):
                self._parse_class_declare(declare_line)

    def _parse_bind_func(self, m_def_line):
        names = re.findall('"(.*?)"', m_def_line)
        if not names:
            return
        func_name = names[0].replace('::', '.')
        if 'py::arg' in m_def_line:
            # deal with m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)", py::arg("input");
            elements = m_def_line.split(',')
            min_args_num = 0
            max_args_name = 0
            for element in elements:
                if not element.strip().startswith('py::arg'):
                    continue
                max_args_name += 1
                if '=' not in element:
                    min_args_num += 1
        else:
            # deal with m.def("forward", &chamfer_forward, "chamfer forward (CUDA)");
            cpp_func_name = m_def_line.split(',')[1].split(')')[0].strip()
            min_args_num, max_args_name = self._parse_cpp_func_args_num(cpp_func_name)
        self._cuda_ops.append(CudaOp(func_name, min_args_num, max_args_name))

    def _parse_cpp_func_args_num(self, cpp_func_name):
        # deal with m.def("get_indice_pairs_2d", &spconv::getIndicePair<2>, "get_indice_pairs_2d");
        cpp_func_name = re.sub('&|\(|\)', '', cpp_func_name).split('<')[0]
        in_func_def_line = False
        func_def_line = ''
        for row, line in enumerate(self._file_lines):
            # escape annotation
            if line.strip().startswith('/'):
                continue
            line = line.split('//')[0].strip()
            # def with c10::Dict<std::string, c10::Dict<std::string, double>> GPUDecoder::
            #     get_metadata() const {xxx}
            if line.endswith('::'):
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
            return MIN_ARGS_NUM, MAX_ARGS_NUM
        if f'{cpp_func_name}()' in func_def_line:
            return MIN_ARGS_NUM, MIN_ARGS_NUM
        return func_def_line.count(',') + 1, func_def_line.count(',') + 1

    def _visit_torch_library_module(self):
        in_torch_library_body = False
        in_func_or_class_declare = False
        declare_line = ''
        declare_lines = []
        for line in self._file_lines:
            # escape annotation
            if line.strip().startswith('/'):
                continue
            line = line.split('//')[0].strip()
            if line.startswith(('TORCH_LIBRARY(', 'TORCH_LIBRARY_FRAGMENT(', 'TORCH_LIBRARY_IMPL(')):
                in_torch_library_body = True
            if not in_torch_library_body:
                continue
            if line.startswith(('m.def(', 'm.impl(', 'm.class')):
                if declare_line:
                    declare_lines.append(declare_line)
                    declare_line = ''
                in_func_or_class_declare = True
            if in_func_or_class_declare:
                declare_line += line
                if line.endswith('}'):
                    declare_lines.append(declare_line)
                    break
        for declare_line in declare_lines:
            if declare_line.startswith('m.def('):
                # m.def(xxx);
                self._parse_torch_library_def(declare_line)
            elif declare_line.startswith('m.impl('):
                # m.impl(xxx);
                self._parse_torch_library_impl(declare_line)
            elif declare_line.startswith('m.class'):
                # m.class xxx
                self._parse_class_declare(declare_line)

    def _parse_torch_library_def(self, func_line):
        if 'TORCH_SELECTIVE_SCHEMA(' in func_line:
            # deal with m.def(TORCH_SELECTIVE_SCHEMA(
            #       "torchvision::roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height
            #       int pooled_width) -> (Tensor, Tensor)"));
            func_name = re.findall('TORCH_SELECTIVE_SCHEMA\("(.*?)\(', func_line)
            if not func_name:
                return
            func_name = func_name[0].replace('::', '.')
            min_args_num = max_args_name = func_line.count(',') + 1
        elif '[](' in func_line:
            # def with m.def("torchaudio::ffmpeg_set_log_level", [](int64_t level) {
            #     av_log_set_level(static_cast<int>(level));
            #   });
            func_name = func_line.split('"')[1].replace('::', '.')
            arg_declare = re.findall('\[\]\((.*?)\)', func_line)
            if not arg_declare:
                return
            arg_declare = arg_declare[0]
            if not arg_declare:
                min_args_num = max_args_name =MIN_ARGS_NUM
            else:
                min_args_num, max_args_name = arg_declare.count(',') + 1
        else:
            # deal with m.def("_cuda_version", &cuda_version);
            # m.def("read_video_from_file", read_video_from_file);
            func_name = func_line.split('"')[1].replace('::', '.')
            cpp_func_name = func_line.split(',')[1].split(')')[0].strip()
            min_args_num, max_args_name = self._parse_cpp_func_args_num(cpp_func_name)
        self._cuda_ops.append(CudaOp(func_name, min_args_num, max_args_name))

    def _parse_torch_library_impl(self, func_line):
        if 'TORCH_SELECTIVE_NAME' in func_line and 'TORCH_FN' in func_line:
            # deal with m.impl(
            #       TORCH_SELECTIVE_NAME("torchvision::roi_align"),
            #       TORCH_FN(qroi_align_forward_kernel));
            func_name = func_line.split('"')[1].replace('::', '.')
            cpp_func_name = re.findall('TORCH_FN\((.*?)\)', func_line)
            if not cpp_func_name:
                return
            cpp_func_name = cpp_func_name[0]
        else:
            # deal with m.impl("rnnt_loss", &compute);
            names = re.findall('"(.*?)"', func_line)
            func_name = names[0].replace('::', '.')
            cpp_func_name = func_line.split(',')[1].split(')')[0].strip()
        min_args_num, max_args_name = self._parse_cpp_func_args_num(cpp_func_name)
        self._cuda_ops.append(CudaOp(func_name, min_args_num, max_args_name))

    def _parse_class_declare(self, func_line):
        # deal with m.class_<GPUDecoder>("GPUDecoder")
        #       .def(torch::init<std::string, torch::Device>())
        #       .def("next", &GPUDecoder::decode);
        names = re.findall('"(.*?)"', func_line)
        if not names:
            return
        class_name = names[0]
        class_init_func = re.search('::init<.*?>', func_line)
        if not class_init_func:
            self._cuda_ops.append(CudaOp(class_name, MIN_ARGS_NUM, MAX_ARGS_NUM))
        else:
            class_init_func = class_init_func.group()
            if '<>' in class_init_func:
                args_name = 0
            else:
                args_name = class_init_func.count(',') + 1
            self._cuda_ops.append(CudaOp(class_name, args_name, args_name))
        if len(names) <= 1:
            return
        for name in names[1:]:
            # instance api ignore args num
            func_name = f'{names[0]}.{name}'.replace('::', '.')
            self._cuda_ops.append(CudaOp(func_name, MIN_ARGS_NUM, MAX_ARGS_NUM))
