import os
import sys
import unittest
from unittest import mock

import google.protobuf.text_format

sys.modules['caffe'] = mock.MagicMock()
sys.modules['caffe.proto'] = mock.MagicMock()
sys.modules['caffe.proto.caffe_pb2'] = mock.MagicMock()

import inplace_layer_process as target_module
from cmp_utils.constant.compare_error import CompareError


class TestCheckInputFileValid(unittest.TestCase):
    def test_valid_input_file(self):
        with mock.patch('os.path.getsize', return_value=1024), \
                mock.patch('cmp_utils.path_check.check_path_valid',
                           return_value=CompareError.MSACCUCMP_NONE_ERROR):
            try:
                target_module.RemoveInplaceLayerProcess._check_input_file_valid('/valid/file.prototxt')
            except Exception:
                self.fail("Should not raise for valid input file")

    def test_input_file_not_exist(self):
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_INVALID_PATH_ERROR):
            with self.assertRaises(CompareError) as ctx:
                target_module.RemoveInplaceLayerProcess._check_input_file_valid('/nonexistent/file.prototxt')
            self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_input_file_too_large(self):
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                mock.patch('os.path.getsize', return_value=target_module.MAX_SIZE + 1):
            with self.assertRaises(CompareError) as ctx:
                target_module.RemoveInplaceLayerProcess._check_input_file_valid('/large/file.prototxt')
            self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_FILE_TOO_LARGE_ERROR)


class TestCheckOutputFileValid(unittest.TestCase):
    def test_valid_output_path(self):
        with mock.patch('os.path.exists', return_value=False), \
                mock.patch('cmp_utils.path_check.check_path_valid',
                           return_value=CompareError.MSACCUCMP_NONE_ERROR):
            try:
                target_module.RemoveInplaceLayerProcess._check_output_file_valid('/valid/output.prototxt')
            except Exception:
                self.fail("Should not raise for valid output path")

    def test_output_path_not_writable(self):
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_INVALID_PATH_ERROR):
            with self.assertRaises(CompareError) as ctx:
                target_module.RemoveInplaceLayerProcess._check_output_file_valid('/invalid/output.prototxt')
            self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_output_path_exists_as_directory(self):
        with mock.patch('os.path.exists', return_value=True), \
                mock.patch('os.path.isfile', return_value=False), \
                mock.patch('cmp_utils.path_check.check_path_valid',
                           return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with self.assertRaises(CompareError) as ctx:
                target_module.RemoveInplaceLayerProcess._check_output_file_valid('/existing/dir')
            self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_SYMLINK_ERROR)

    def test_output_path_exists_as_file(self):
        with mock.patch('os.path.exists', return_value=True), \
                mock.patch('os.path.isfile', return_value=True), \
                mock.patch('os.remove'), \
                mock.patch('cmp_utils.path_check.check_path_valid',
                           return_value=CompareError.MSACCUCMP_NONE_ERROR):
            try:
                target_module.RemoveInplaceLayerProcess._check_output_file_valid('/existing/file.prototxt')
            except Exception:
                self.fail("Should not raise when existing file is removed")


class TestInit(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv

    def tearDown(self):
        sys.argv = self._orig_argv

    def test_init_with_output_path(self):
        sys.argv = ['prog', '-i', '/input/model.prototxt', '-o', '/output/model.prototxt']
        with mock.patch('os.path.realpath', side_effect=lambda x: x):
            obj = target_module.RemoveInplaceLayerProcess()
            self.assertEqual(obj.input_file_path, '/input/model.prototxt')
            self.assertEqual(obj.output_file_path, '/output/model.prototxt')
            self.assertIsNone(obj.net_param)
            self.assertEqual(obj.cur_layer_idx, -1)

    def test_init_without_output_path(self):
        sys.argv = ['prog', '-i', '/input/model.prototxt']
        with mock.patch('os.path.realpath', side_effect=lambda x: x), \
                mock.patch('os.path.dirname', return_value='/input'), \
                mock.patch('os.path.basename', return_value='model.prototxt'), \
                mock.patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            obj = target_module.RemoveInplaceLayerProcess()
            self.assertEqual(obj.input_file_path, '/input/model.prototxt')
            self.assertEqual(obj.output_file_path, '/input/new_model.prototxt')
            self.assertIsNone(obj.net_param)
            self.assertEqual(obj.cur_layer_idx, -1)


class TestCheckArgumentsValid(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv

    def tearDown(self):
        sys.argv = self._orig_argv

    def test_check_arguments_valid(self):
        sys.argv = ['prog', '-i', '/input/model.prototxt', '-o', '/output/model.prototxt']
        with mock.patch('os.path.realpath', side_effect=lambda x: x):
            obj = target_module.RemoveInplaceLayerProcess()
            with mock.patch.object(obj, '_check_input_file_valid'), \
                    mock.patch.object(obj, '_check_output_file_valid'):
                try:
                    obj.check_arguments_valid()
                except Exception:
                    self.fail("Should not raise for valid arguments")


class TestHandleTop(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv

    def tearDown(self):
        sys.argv = self._orig_argv

    def _make_obj(self):
        sys.argv = ['prog', '-i', '/input/model.prototxt']
        with mock.patch('os.path.realpath', side_effect=lambda x: x), \
                mock.patch('os.path.dirname', return_value='/input'), \
                mock.patch('os.path.basename', return_value='model.prototxt'), \
                mock.patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            return target_module.RemoveInplaceLayerProcess()

    def test_handle_top_dropout_single_bottom_top_ne_bottom(self):
        obj = self._make_obj()
        layer_item = mock.MagicMock()
        layer_item.type = 'Dropout'
        layer_item.bottom = ['input_data']
        layer_item.top = ['dropout_output']
        ok, old_name, new_name = obj._handle_top(layer_item, 0)
        self.assertTrue(ok)
        self.assertEqual(old_name, 'dropout_output')
        self.assertEqual(new_name, 'input_data')

    def test_handle_top_dropout_multi_bottom(self):
        obj = self._make_obj()
        layer_item = mock.MagicMock()
        layer_item.type = 'Dropout'
        layer_item.bottom = ['input1', 'input2']
        layer_item.top = ['dropout_output']
        ok, old_name, new_name = obj._handle_top(layer_item, 0)
        self.assertTrue(ok)
        self.assertEqual(old_name, '')
        self.assertEqual(new_name, '')

    def test_handle_top_inplace_layer(self):
        obj = self._make_obj()
        layer_item = mock.MagicMock()
        layer_item.type = 'ReLU'
        layer_item.name = 'relu1'
        layer_item.top = ['old_name']
        layer_item.bottom = ['old_name']
        ok, old_name, new_name = obj._handle_top(layer_item, 0)
        self.assertTrue(ok)
        self.assertEqual(old_name, 'old_name')
        self.assertEqual(new_name, 'relu1')

    def test_handle_top_inplace_multi_top(self):
        obj = self._make_obj()
        layer_item = mock.MagicMock()
        layer_item.type = 'ReLU'
        layer_item.name = 'relu1'
        layer_item.top = ['old_name', 'extra_top']
        layer_item.bottom = ['old_name']
        ok, old_name, new_name = obj._handle_top(layer_item, 0)
        self.assertFalse(ok)

    def test_handle_top_inplace_multi_bottom(self):
        obj = self._make_obj()
        layer_item = mock.MagicMock()
        layer_item.type = 'ReLU'
        layer_item.name = 'relu1'
        layer_item.top = ['old_name']
        layer_item.bottom = ['old_name', 'extra_bottom']
        ok, old_name, new_name = obj._handle_top(layer_item, 0)
        self.assertFalse(ok)

    def test_handle_top_top_equals_name(self):
        obj = self._make_obj()
        layer_item = mock.MagicMock()
        layer_item.type = 'ReLU'
        layer_item.name = 'relu1'
        layer_item.top = ['relu1']
        layer_item.bottom = ['input_data']
        ok, old_name, new_name = obj._handle_top(layer_item, 0)
        self.assertFalse(ok)


class TestFindName(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv

    def tearDown(self):
        sys.argv = self._orig_argv

    def _make_obj(self):
        sys.argv = ['prog', '-i', '/input/model.prototxt']
        with mock.patch('os.path.realpath', side_effect=lambda x: x), \
                mock.patch('os.path.dirname', return_value='/input'), \
                mock.patch('os.path.basename', return_value='model.prototxt'), \
                mock.patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            return target_module.RemoveInplaceLayerProcess()

    def test_find_name_found(self):
        obj = self._make_obj()
        layer = mock.MagicMock()
        layer.type = 'ReLU'
        layer.name = 'relu1'
        layer.top = ['old_name']
        layer.bottom = ['old_name']
        obj.net_param = mock.MagicMock()
        obj.net_param.layer = [layer]
        old_name, new_name = obj._find_name()
        self.assertEqual(old_name, 'old_name')
        self.assertEqual(new_name, 'relu1')

    def test_find_name_not_found(self):
        obj = self._make_obj()
        layer = mock.MagicMock()
        layer.type = 'ReLU'
        layer.name = 'relu1'
        layer.top = ['relu1']
        layer.bottom = ['input_data']
        obj.net_param = mock.MagicMock()
        obj.net_param.layer = [layer]
        old_name, new_name = obj._find_name()
        self.assertEqual(old_name, '')
        self.assertEqual(new_name, '')

    def test_find_name_skip_processed_layers(self):
        obj = self._make_obj()
        layer1 = mock.MagicMock()
        layer1.type = 'ReLU'
        layer1.name = 'relu1'
        layer1.top = ['relu1']
        layer1.bottom = ['input_data']
        layer2 = mock.MagicMock()
        layer2.type = 'ReLU'
        layer2.name = 'relu2'
        layer2.top = ['old_name']
        layer2.bottom = ['old_name']
        obj.net_param = mock.MagicMock()
        obj.net_param.layer = [layer1, layer2]
        obj.cur_layer_idx = 0
        old_name, new_name = obj._find_name()
        self.assertEqual(old_name, 'old_name')
        self.assertEqual(new_name, 'relu2')


class TestParseName(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv

    def tearDown(self):
        sys.argv = self._orig_argv

    def _make_obj(self):
        sys.argv = ['prog', '-i', '/input/model.prototxt']
        with mock.patch('os.path.realpath', side_effect=lambda x: x), \
                mock.patch('os.path.dirname', return_value='/input'), \
                mock.patch('os.path.basename', return_value='model.prototxt'), \
                mock.patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            return target_module.RemoveInplaceLayerProcess()

    def test_parse_name_renames_bottom_and_top(self):
        obj = self._make_obj()
        layer = mock.MagicMock()
        layer.bottom = ['old_name', 'other_name']
        layer.top = ['old_name']
        obj.net_param = mock.MagicMock()
        obj.net_param.layer = [layer]
        obj._parse_name('old_name', 'new_name')
        self.assertEqual(layer.bottom[0], 'new_name')
        self.assertEqual(layer.top[0], 'new_name')

    def test_parse_name_skips_processed_layers(self):
        obj = self._make_obj()
        layer1 = mock.MagicMock()
        layer1.bottom = ['old_name']
        layer1.top = ['old_name']
        layer2 = mock.MagicMock()
        layer2.bottom = ['old_name']
        layer2.top = ['old_name']
        obj.net_param = mock.MagicMock()
        obj.net_param.layer = [layer1, layer2]
        obj.cur_layer_idx = 0
        obj._parse_name('old_name', 'new_name')
        self.assertEqual(layer1.bottom[0], 'old_name')
        self.assertEqual(layer2.bottom[0], 'new_name')

    def test_parse_name_no_match(self):
        obj = self._make_obj()
        layer = mock.MagicMock()
        layer.bottom = ['other_name']
        layer.top = ['other_name']
        obj.net_param = mock.MagicMock()
        obj.net_param.layer = [layer]
        obj._parse_name('old_name', 'new_name')
        self.assertEqual(layer.bottom[0], 'other_name')
        self.assertEqual(layer.top[0], 'other_name')


class TestRemoveInplaceLayer(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv

    def tearDown(self):
        sys.argv = self._orig_argv

    def _make_obj(self):
        sys.argv = ['prog', '-i', '/input/model.prototxt']
        with mock.patch('os.path.realpath', side_effect=lambda x: x), \
                mock.patch('os.path.dirname', return_value='/input'), \
                mock.patch('os.path.basename', return_value='model.prototxt'), \
                mock.patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            return target_module.RemoveInplaceLayerProcess()

    @mock.patch('inplace_layer_process.caffe_pb2')
    @mock.patch('inplace_layer_process.google.protobuf.text_format')
    def test_remove_inplace_normal(self, mock_text_format, mock_caffe_pb2):
        obj = self._make_obj()
        mock_net = mock.MagicMock()
        mock_caffe_pb2.NetParameter.return_value = mock_net

        layer1 = mock.MagicMock()
        layer1.type = 'ReLU'
        layer1.name = 'relu1'
        layer1.top = ['relu1']
        layer1.bottom = ['input_data']
        layer2 = mock.MagicMock()
        layer2.type = 'Dropout'
        layer2.name = 'dropout1'
        layer2.top = ['dropout_output']
        layer2.bottom = ['input_data']
        mock_net.layer = [layer1, layer2]

        mock_open_func = mock.mock_open()
        mock_fd = mock.MagicMock()
        with mock.patch('builtins.open', mock_open_func), \
                mock.patch('os.open', return_value=3), \
                mock.patch('os.fdopen', return_value=mock_fd), \
                mock.patch.object(obj, 'check_arguments_valid'), \
                mock.patch.object(obj, '_find_name', side_effect=[('', '')]):
            obj.remove_inplace_layer()

    @mock.patch('inplace_layer_process.caffe_pb2')
    @mock.patch('inplace_layer_process.google.protobuf.text_format')
    def test_remove_inplace_with_inplace_rename(self, mock_text_format, mock_caffe_pb2):
        obj = self._make_obj()
        mock_net = mock.MagicMock()
        mock_caffe_pb2.NetParameter.return_value = mock_net

        layer1 = mock.MagicMock()
        layer1.type = 'ReLU'
        layer1.name = 'relu1'
        layer1.top = ['old_name']
        layer1.bottom = ['old_name']
        layer2 = mock.MagicMock()
        layer2.type = 'Dropout'
        layer2.name = 'dropout1'
        layer2.top = ['dropout_output']
        layer2.bottom = ['input_data']
        mock_net.layer = [layer1, layer2]

        mock_fd = mock.MagicMock()
        with mock.patch('builtins.open', mock.mock_open()), \
                mock.patch('os.open', return_value=3), \
                mock.patch('os.fdopen', return_value=mock_fd), \
                mock.patch.object(obj, 'check_arguments_valid'), \
                mock.patch.object(obj, '_find_name', side_effect=[('old_name', 'relu1'), ('', '')]), \
                mock.patch.object(obj, '_parse_name'):
            obj.remove_inplace_layer()

    @mock.patch('inplace_layer_process.caffe_pb2')
    @mock.patch('google.protobuf.text_format.Parse')
    def test_remove_inplace_parse_error(self, mock_parse, mock_caffe_pb2):
        obj = self._make_obj()
        mock_caffe_pb2.NetParameter.return_value = mock.MagicMock()

        parse_error = google.protobuf.text_format.ParseError('parse error')
        mock_parse.side_effect = parse_error

        with mock.patch('builtins.open', mock.mock_open()), \
                mock.patch.object(obj, 'check_arguments_valid'):
            with self.assertRaises(CompareError) as ctx:
                obj.remove_inplace_layer()
            self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    @mock.patch('inplace_layer_process.caffe_pb2')
    @mock.patch('inplace_layer_process.google.protobuf.text_format')
    def test_remove_inplace_no_dropout(self, mock_text_format, mock_caffe_pb2):
        obj = self._make_obj()
        mock_net = mock.MagicMock()
        mock_caffe_pb2.NetParameter.return_value = mock_net

        layer1 = mock.MagicMock()
        layer1.type = 'ReLU'
        layer1.name = 'relu1'
        layer1.top = ['relu1']
        layer1.bottom = ['input_data']
        mock_net.layer = [layer1]

        mock_fd = mock.MagicMock()
        with mock.patch('builtins.open', mock.mock_open()), \
                mock.patch('os.open', return_value=3), \
                mock.patch('os.fdopen', return_value=mock_fd), \
                mock.patch.object(obj, 'check_arguments_valid'), \
                mock.patch.object(obj, '_find_name', side_effect=[('', '')]):
            obj.remove_inplace_layer()

    @mock.patch('inplace_layer_process.caffe_pb2')
    @mock.patch('google.protobuf.text_format.Parse')
    def test_remove_inplace_unicode_decode_error(self, mock_parse, mock_caffe_pb2):
        obj = self._make_obj()
        mock_caffe_pb2.NetParameter.return_value = mock.MagicMock()

        mock_parse.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')

        with mock.patch('builtins.open', mock.mock_open()), \
                mock.patch.object(obj, 'check_arguments_valid'):
            with self.assertRaises(CompareError) as ctx:
                obj.remove_inplace_layer()
            self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    @mock.patch('inplace_layer_process.caffe_pb2')
    @mock.patch('inplace_layer_process.google.protobuf.text_format')
    def test_remove_inplace_with_dropout_removal(self, mock_text_format, mock_caffe_pb2):
        obj = self._make_obj()
        mock_net = mock.MagicMock()
        mock_caffe_pb2.NetParameter.return_value = mock_net

        layer1 = mock.MagicMock()
        layer1.type = 'Dropout'
        layer1.name = 'dropout1'
        layer1.top = ['dropout_output']
        layer1.bottom = ['dropout_input']
        mock_net.layer = mock.MagicMock()
        mock_net.layer.__iter__.return_value = iter([layer1])

        mock_fd = mock.MagicMock()
        with mock.patch('builtins.open', mock.mock_open()), \
                mock.patch('os.open', return_value=3), \
                mock.patch('os.fdopen', return_value=mock_fd), \
                mock.patch.object(obj, 'check_arguments_valid'), \
                mock.patch.object(obj, '_find_name', side_effect=[('', '')]):
            obj.remove_inplace_layer()
            mock_net.layer.remove.assert_called_once_with(layer1)


if __name__ == '__main__':
    unittest.main()