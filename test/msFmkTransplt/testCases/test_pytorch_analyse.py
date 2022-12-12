#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os
import shutil
import sys
import unittest
import unittest.mock as mock

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))

ANALYSE_ERROR = 1


class Args:
    def __init__(self, input_path, output_path, version='1.8.1', mode='torch_apis'):
        self.input = input_path
        self.output = output_path
        self.version = version
        self.mode = mode


def run(mock_args):
    from analysis.analyse import PyTorchAnalyse
    from src.ms_fmk_transplt.utils import trans_utils as utils
    try:
        utils.refresh_parso_cache = mock.Mock(side_effect=mock_refresh_parso_cache())
        analyse = PyTorchAnalyse()
        analyse._PyTorchAnalyse__parse_command = mock_args
        return analyse.main()
    except Exception as exp:
        print(repr(exp))
        return ANALYSE_ERROR


def mock_refresh_parso_cache():
    pass


class TestPyTorchAnalyse(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.ms_fmk_transplt.utils import trans_utils as utils
        utils.IS_JEDI_INSTALLED = True

    def setUp(self):
        self.abs_input_path = os.path.abspath('../resources/net')
        shutil.rmtree("../test_result/", ignore_errors=True)
        os.makedirs("../test_result/analyse_result", exist_ok=True)
        self.abs_output_path = os.path.join(os.path.abspath("../test_result"), "analyse_result")
        self.has_error = False

    def test_analysis(self):
        mock_args = mock.Mock(return_value=Args(os.path.join(self.abs_input_path, "barlowtwins_amp"),
                                                self.abs_output_path))

        self.assertNotEqual(run(mock_args), ANALYSE_ERROR)

        mock_args = mock.Mock(return_value=Args(os.path.join(self.abs_input_path, "ID0329_CarPeting_Pytorch_FD-GAN"),
                                                self.abs_output_path, mode='third_party'))

        self.assertNotEqual(run(mock_args), ANALYSE_ERROR)

    def test_cuda_op_parser(self):
        from analysis.third_party_analysis.cuda_cpp_visitor import CudaOpVisitor
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

    @classmethod
    def tearDownClass(cls) -> None:
        cuda_op_project_path = './cuda_op_test'
        shutil.rmtree(cuda_op_project_path, ignore_errors=True)
