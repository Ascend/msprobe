import torch
import pandas as pd
import unittest
from unittest import TestCase, mock
from unittest.mock import patch

from msprobe.core.common.const import MonitorConst
from msprobe.core.monitor.anomaly_processor import AnomalyTurbulence
from msprobe.pytorch.monitor.data_writers import BaseWriterWithAD, CSVWriterWithAD, SummaryWriterWithAD, WriterInput


class TestBaseWriterWithAD(TestCase):

    def setUp(self) -> None:
        self.writer = BaseWriterWithAD(WriterInput('', None, None))

    def test_stack_tensors(self):
        tensors = [torch.tensor(1.0), torch.tensor(2.0)]
        result = self.writer.stack_tensors(tensors)

        self.assertEqual(len(result), 2)
        for i, t in enumerate(result):
            self.assertTrue(torch.allclose(t, tensors[i].cpu()))

    def test_get_anomalies(self):
        expected = []

        self.assertEqual(self.writer.get_anomalies(), expected)

    def test_clear_anomalies(self):
        self.writer.anomalies = ['anomaly1', 'anomaly2']
        self.writer.clear_anomalies()

        self.assertEqual(self.writer.anomalies, [])

    @patch("msprobe.pytorch.monitor.data_writers.logger")
    def test_add_scalar(self, mock_logger):
        AnomalyTurbulence_obj = AnomalyTurbulence(0.2)
        self.writer.ad_rules = [AnomalyTurbulence_obj]
        tag = ('0:1.post_attention_norm.weight/rank0/pre_grad', 'mean')
        self.writer.tag2scalars = {tag: {'avg': 1.0, 'count': 1}}
        self.writer.add_scalar(tag, 2.0)

        mock_logger.info.assert_called_once()

    @patch.object(BaseWriterWithAD, "stack_tensors")
    @patch.object(BaseWriterWithAD, "add_scalar")
    def test_write_metrics_empty_metric_value(self, mock_add_scalar, mock_stack_tensors):
        metric_value = {}
        self.writer.write_metrics(ops=['op1'], metric_value=metric_value, step=1)
        mock_add_scalar.assert_not_called()
        mock_stack_tensors.assert_not_called()

    @patch.object(BaseWriterWithAD, "stack_tensors")
    @patch.object(BaseWriterWithAD, "add_scalar")
    def test_write_metrics_empty_tensors(self, mock_add_scalar, mock_stack_tensors):
        """metric_value不空但内部没tensor时应return"""
        metric_value = {'key1': {}}  # 内部values为空
        self.writer.write_metrics(ops=['op1'], metric_value=metric_value, step=1)
        mock_add_scalar.assert_not_called()
        mock_stack_tensors.assert_not_called()

    @patch.object(BaseWriterWithAD, "stack_tensors", side_effect=lambda x: x)
    @patch.object(BaseWriterWithAD, "add_scalar")
    def test_write_metrics_normal(self, mock_add_scalar, mock_stack_tensors):
        """正常路径，应调用stack_tensors和add_scalar"""
        metric_value = {
            'layer1': {'grad': torch.tensor(1.0), 'act': torch.tensor(2.0)},
            'layer2': {'grad': torch.tensor(3.0), 'act': torch.tensor(4.0)}
        }
        ops = ['grad', 'act']
        step = 10
        self.writer.write_metrics(ops, metric_value, step)
        # 检查stack_tensors调用次数与传入切片
        expected_total_tensors = len(metric_value) * len(ops)
        n_calls = (expected_total_tensors + MonitorConst.SLICE_SIZE - 1) // MonitorConst.SLICE_SIZE
        self.assertEqual(mock_stack_tensors.call_count, n_calls)
        # 检查add_scalar调用次数与tag格式
        self.assertEqual(mock_add_scalar.call_count, expected_total_tensors)

    def test_ad(self):
        AnomalyTurbulence_obj = AnomalyTurbulence(0.2)
        self.writer.ad_rules = [AnomalyTurbulence_obj]
        expected = True, "AnomalyTurbulence"

        self.assertEqual(self.writer._ad(2.0, 1.0), expected)

    def test_update_tag2scalars(self):
        self.writer._update_tag2scalars('tag1', 1.0)
        self.assertEqual(self.writer.tag2scalars['tag1']['avg'], 1.0)
        self.assertEqual(self.writer.tag2scalars['tag1']['count'], 1)
        self.writer._update_tag2scalars('tag1', 2.0)
        self.assertEqual(self.writer.tag2scalars['tag1']['avg'], 1.01)
        self.assertEqual(self.writer.tag2scalars['tag1']['count'], 2)


class TestCsvWriterWithAD(TestCase):
    def setUp(self) -> None:
        with mock.patch("msprobe.pytorch.monitor.data_writers.create_directory"):
            self.writer = CSVWriterWithAD(WriterInput('', None, None, step_count_per_record=1))

    def test_get_step_interval(self):
        step = 5
        self.assertEqual(self.writer.get_step_interval(step), (5, 5))

    @mock.patch("msprobe.pytorch.monitor.data_writers.write_df_to_csv")
    def test_write_csv_empty_context(self, mock_write):
        self.writer.write_csv("reduced_grad", 10)
        mock_write.assert_not_called()

    @mock.patch("msprobe.pytorch.monitor.data_writers.write_df_to_csv")
    @mock.patch("msprobe.pytorch.monitor.data_writers.os.path.exists", return_value=False)
    def test_write_csv_normal(self, mock_exists, mock_write):
        self.writer.context_dict = {"0:layer.0.weight": [0.2, 0, -0.2]}
        self.writer.header = ["vpp_stage", "name", "step", "max", "mean", "min"]
        self.writer.write_csv("reduced_grad", 10)

        # 第一次调用写 header
        self.assertIsInstance(mock_write.call_args_list[0][0][0], pd.DataFrame)
        # 第二次调用写内容
        df_written = mock_write.call_args_list[1][0][0]
        self.assertTrue("layer.0.weight" in df_written.iloc[0, 1])
        # 写完应清空 context_dict
        self.assertEqual(len(self.writer.context_dict), 0)

    def test_add_scalar_tensor_float_size(self):
        tag = ("layer0.weight/rank0/pre_grad", "min")
        # tensor
        self.writer.add_scalar(tag, torch.tensor(2.8), 5)
        self.assertIn("layer0.weight", self.writer.context_dict)
        self.assertAlmostEqual(self.writer.context_dict["layer0.weight"][0], 2.8, places=3)

        # float
        self.writer.add_scalar(tag, 2.8, 6)
        self.assertEqual(self.writer.context_dict["layer0.weight"][-1], 2.8)

        # torch.Size
        self.writer.add_scalar(tag, torch.Size([2, 3]), 7)
        self.assertEqual(self.writer.context_dict["layer0.weight"][-1], [2, 3])

    def test_close(self):
        self.writer.close()  # 不应抛异常


if __name__ == '__main__':
    unittest.main()