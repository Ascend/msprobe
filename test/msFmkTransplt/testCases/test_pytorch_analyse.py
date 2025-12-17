#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import os
import shutil
import sys
import unittest
import unittest.mock as mock
from unittest.mock import MagicMock
import torch
from libcst import parse_statement, Call, Name, Module, SimpleStatementLine, IndentedBlock, Pass, FunctionDef, \
    Parameters
import libcst.matchers as m

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))

ANALYSE_ERROR = 1


class Args:
    def __init__(self, input_path, output_path, version='2.1.0', mode='torch_apis', api_files='', env_path=None):
        self.input = input_path
        self.output = output_path
        self.version = version
        self.mode = mode
        self.api_files = api_files.split()
        self.env_path = env_path


def run(mock_args):
    from analysis.pytorch_analyse import PyTorchAnalyse
    try:
        analyse = PyTorchAnalyse()
        analyse._PyTorchAnalyse__parse_command = mock_args
        return analyse.main()
    except Exception as exp:
        print(repr(exp))
        return ANALYSE_ERROR


class TestPyTorchAnalyse(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cuda_op_project_path = './cuda_op_test'
        shutil.rmtree(cuda_op_project_path, ignore_errors=True)

    def setUp(self):
        self.abs_input_path = os.path.abspath('../resources/net')
        self.abs_api_files_path = os.path.abspath('../resources/api_files')
        shutil.rmtree("../test_result/", ignore_errors=True)
        os.makedirs("../test_result/analyse_result", exist_ok=True)
        self.abs_output_path = os.path.join(os.path.abspath("../test_result"), "analyse_result")
        self.has_error = False

    def test_analysis(self):
        mock_args = mock.Mock(return_value=Args(os.path.join(self.abs_input_path, "barlowtwins_amp"),
                                                self.abs_output_path, api_files=
                                                os.path.join(self.abs_api_files_path, "3rd_party_unsupported_api.csv")))

        self.assertNotEqual(run(mock_args), ANALYSE_ERROR)

        mock_args = mock.Mock(return_value=Args(os.path.join(self.abs_input_path, "ID0329_CarPeting_Pytorch_FD-GAN"),
                                                self.abs_output_path, mode='third_party'))

        self.assertNotEqual(run(mock_args), ANALYSE_ERROR)

    def test_cuda_op_parser(self):
        from analysis.unsupported_api_analysis.cuda_cpp_visitor import CudaOpVisitor
        from src.ms_fmk_transplt.utils import trans_utils as utils
        code = '''
int chamfer_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2) {
    return chamfer_cuda_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);
}
int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}
// pybind11_module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def
  m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)", py::arg("input"),
        py::arg("kernel"), py::arg("up_x"), py::arg("up_y"), py::arg("down_x"),
        py::arg("down_y"), py::arg("pad_x0")=1, py::arg("pad_x1"),
        py::arg("pad_y0"), py::arg("pad_y1"));
  m.def("forward", &chamfer_forward, "chamfer forward (CUDA)");

  // py_class
  py::class_<StreamWriterFileObj, c10::intrusive_ptr<StreamWriterFileObj>>(
      m, "StreamWriterFileObj")
      .def(py::init<py::object, const c10::optional<std::string>&, int64_t>())
      .def("set_metadata", &StreamWriterFileObj::set_metadata)
      .def("add_audio_stream", &StreamWriterFileObj::add_audio_stream)
      .def("add_video_stream", &StreamWriterFileObj::add_video_stream)
      .def("dump_format", &StreamWriterFileObj::dump_format)
      .def("open", &StreamWriterFileObj::open)
      .def("write_audio_chunk", &StreamWriterFileObj::write_audio_chunk)
      .def("write_video_chunk", &StreamWriterFileObj::write_video_chunk)
      .def("flush", &StreamWriterFileObj::flush)
      .def("close", &StreamWriterFileObj::close);
}

// torch library
TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
    // m.def
    m.def(
      "torchaudio::_lfilter(Tensor waveform, Tensor a_coeffs, Tensor b_coeffs) -> Tensor");
    m.def(TORCH_SELECTIVE_SCHEMA(
          "torchvision::ps_roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)"));
    m.def("torchaudio::ffmpeg_set_log_level", [](int64_t level) {
        av_log_set_level(static_cast<int>(level));
      });
    m.def("_cuda_version", &cuda_version);

    // m.impl
    m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
    m.impl("rnnt_loss", rnnt_loss_autograd);

    // m.class
    m.class_<GPUDecoder>("GPUDecoder")
      .def(torch::init<std::string, torch::Device>())
      .def("seek", &GPUDecoder::seek)
      .def("get_metadata", &GPUDecoder::get_metadata)
      .def("next", &GPUDecoder::decode);
 }
        '''
        project_path = './cuda_op_test'
        shutil.rmtree(project_path, ignore_errors=True)
        os.makedirs(project_path, exist_ok=True)
        utils.write_file_content(os.path.join(project_path, 'cuda.cpp'), code)
        cuda_op_visitor = CudaOpVisitor(project_path)
        cuda_op_visitor.visit_cuda_files()
        cuda_op_list = cuda_op_visitor.cuda_ops
        self.assertEqual(len(cuda_op_list), 22)
        self.assertEqual(cuda_op_list[2].max_args_num, 2)
        self.assertEqual(cuda_op_list[12].max_args_num, 3)

    def test_dynamic_shape_analysis_hook(self):
        from analysis.dynamic_shape_analysis.msft_dynamic_analysis.hook import Logger, ShapeRange, TraceInfo, \
            DynamicShapeDetect
        test_func = "test_func"
        Logger()
        # ShapeRange
        shape_range = ShapeRange()
        shape_range.update((1, 2, 3))
        self.assertEqual(str(shape_range), '[(1, 2, 3)-(1, 2, 3)]')
        shape_range.update((1, 2, 4))
        self.assertEqual(str(shape_range), '[(1, 2, 3)-(1, 2, 4)]')
        shape_range.update((1, 3, 5))
        self.assertEqual(str(shape_range), '[(1, 2, 3)-(1, 3, 5)]')
        shape_range.update((1, 3, 4))
        self.assertEqual(str(shape_range), '[(1, 2, 3)-(1, 3, 5)]')
        # TraceInfo
        trace_info = TraceInfo(None, "test_api", 1)
        input_shape_list = [(2, 3), (4, 5)]
        trace_info.update_input_shape_range(input_shape_list)
        self.assertEqual(len(trace_info.input_shape_range), 2)
        self.assertEqual(trace_info.input_shape_range[0].max_shape_len, 2)
        self.assertEqual(trace_info.input_shape_range[1].max_shape_len, 2)
        # DynamicShapeDetect hook_func
        dsd = DynamicShapeDetect()
        dsd.hook_func = MagicMock(return_value=torch.tensor([1, 2, 3]))
        result = dsd.hook_func(torch.tensor([1, 2, 3]), test_func, 1, a=torch.tensor([1, 2, 3]))
        self.assertEqual(result.tolist(), [1, 2, 3])
        # DynamicShapeDetect start
        dataset = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        for data in dsd.start(dataset):
            self.assertIsInstance(data, torch.Tensor)
        # DynamicShapeDetect after_call
        key = 'test_trace_test_func_0_0'
        result = torch.tensor([1, 2, 3])
        dsd.trace_dict[key] = TraceInfo([], test_func, 0)
        dsd.unique_trace_dict['test_trace_test_func_0'] = TraceInfo([], test_func, 0)
        dsd._after_call(key, result)
        self.assertEqual(len(dsd.dynamic_api_set), 0)
        # DynamicShapeDetect before_call
        key = 'test_trace_test_func_0_0'
        args = [torch.tensor([1, 2, 3])]
        kwargs = {}
        dsd.trace_dict[key] = TraceInfo([], test_func, 0)
        dsd.unique_trace_dict[key] = TraceInfo([], test_func, 0)
        dsd._before_call(key, torch.add, args, kwargs)
        self.assertEqual(len(dsd.dynamic_api_set), 0)
        # DynamicShapeDetect get_trace_info_key
        func_name = test_func
        trace = []
        call_number = 0
        key = dsd._get_trace_info_key(func_name, trace, call_number)
        self.assertEqual(key, f'{trace}_{func_name}_{call_number}_0')

    def test_dynamic_shape_converter(self):
        from analysis.dynamic_shape_analysis.dynamic_shape_converter import DynamicShapeTransformer
        transformer = DynamicShapeTransformer()
        body_item = parse_statement("import os")
        self.assertFalse(transformer.verify_import_position(body_item))

        function_body = IndentedBlock(
            body=[
                SimpleStatementLine(body=[Pass()])
            ]
        )
        function_def = FunctionDef(
            name=Name("my_function"),
            params=Parameters(),
            body=function_body
        )
        module = Module(
            body=[
                function_def
            ]
        )
        original_node = module
        updated_node = transformer.leave_Module(original_node, original_node)
        self.assertIn("from msft_dynamic_analysis.hook import DETECTOR", updated_node.code)

        original_node = Call(func=Name("print"))
        result = transformer._check_if_need_hook(original_node)
        self.assertFalse(result)

        node = function_def
        result = transformer._get_parent_node(node, m.FunctionDef())
        self.assertEqual(result, node)
