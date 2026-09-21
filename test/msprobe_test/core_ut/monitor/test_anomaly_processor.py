# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock

from msprobe.core.monitor.anomaly_processor import ScanRule, AnomalyTurbulence, AnomalyNan, AnomalyScanner, \
    AnomalyDataFactory, GradAnomalyData, AnomalyDataWriter, AnomalyDataLoader, AnomalyAnalyse, \
    _get_step_and_stop, _anomaly_analyse, _get_parse_args


class TestScanRule(TestCase):
    def test_apply_not_implemented(self):
        scan_rule = ScanRule()
        with self.assertRaises(Exception) as context:
            scan_rule.apply(None, None)

        self.assertEqual(str(context.exception), "abstract method apply is not implemented")


class TestAnomalyTurbulence(TestCase):

    def setUp(self) -> None:
        self.threshold = 0.2
        self.rule = AnomalyTurbulence(self.threshold)

    def test_apply_with_positive_baseline(self):
        history = 12
        cur = 16
        result = self.rule.apply(cur, history=history)
        self.assertTrue(result)

    def test_apply_with_non_positive_baseline(self):
        history = 0
        cur = -1
        result = self.rule.apply(cur, history=history)
        self.assertTrue(result)

    def test_apply_with_valid_value(self):
        history = 0
        cur = 0
        result = self.rule.apply(cur, history=history)
        self.assertFalse(result)


class TestAnomalyNan(TestCase):

    def setUp(self) -> None:
        self.threshold = 1e10
        self.rule = AnomalyNan(self.threshold)

    def test_apply_with_nan(self):
        cur = float("nan")
        result = self.rule.apply(cur)
        self.assertTrue(result)

    def test_apply_with_big_value(self):
        cur = float("1e30")
        result = self.rule.apply(cur)
        self.assertTrue(result)

    def test_apply_with_valid_value(self):
        cur = 0.5
        result = self.rule.apply(cur)
        self.assertFalse(result)


class TestAnomalyScanner(TestCase):

    def test_load_rules_with_valied_spec(self):
        specs = [
            {"rule_name": "AnomalyTurbulence", "args": {"threshold": 0.2}}
        ]
        rules = AnomalyScanner.load_rules(specs)

        self.assertEqual(len(rules), 1)
        self.assertIsInstance(rules[0], AnomalyTurbulence)
        self.assertEqual(rules[0].threshold, 0.2)

        rules = AnomalyScanner.load_rules(None)
        self.assertEqual(len(rules), 0)

    @patch("msprobe.core.monitor.anomaly_processor.logger")
    def test_load_rules_with_missing_keys(self, mock_logger):
        specs = [
            {"rule_name": "AnomalyTurbulence"}
        ]
        rules = AnomalyScanner.load_rules(specs)

        self.assertEqual(len(rules), 0)
        mock_logger.warning.assert_called_once_with(f"Spec is missing required keys: {specs[0]}")

    def test_load_rules_with_invalid_rule(self):
        # test invalid rule_name
        specs = [{"rule_name": "InvalidRule", "args": {"threshold": 0.2}}]
        rules = AnomalyScanner.load_rules(specs)
        self.assertEqual(len(rules), 0)

        # test invalid args
        specs = [{"rule_name": "AnomalyTurbulence", "args": "invalid args"}]
        rules = AnomalyScanner.load_rules(specs)
        self.assertEqual(len(rules), 0)

    def test_scan(self):
        ad_rules = [AnomalyTurbulence(0.2)]
        # test scan with anomaly
        expected = True, "AnomalyTurbulence"
        self.assertEqual(AnomalyScanner.scan(ad_rules, 1.0, 2.0), expected)
        # test scan with no anomaly
        expected = False, None
        self.assertEqual(AnomalyScanner.scan(ad_rules, 1.0, 1.0), expected)


class TestAnomalyDataFactory(TestCase):

    def setUp(self) -> None:
        rank = 0
        pp_stage = 0
        group_mates = [0]
        self.AnomalyDataFactory = AnomalyDataFactory(rank, pp_stage, group_mates)

    def test_set_call_id(self):
        name2callid = {'param_name': 0}
        self.AnomalyDataFactory.set_call_id(name2callid)

        self.assertEqual(self.AnomalyDataFactory.name2callid, {'param_name': 0})

    def test_create_success(self):
        tag = ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min')
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2."
        step = 2
        result = self.AnomalyDataFactory.create(tag, message, step)

        self.assertEqual(result.step, step)
        self.assertEqual(result.tag_name, tag[0])
        self.assertEqual(result.message, message)
        self.assertEqual(result.vpp_stage, 0)

        # test no vpp_stage
        tag = ('1.self_attention.core_attention_flash_0/rank0/output', 'min')
        result = self.AnomalyDataFactory.create(tag, message, step)
        self.assertEqual(result.vpp_stage, 0)

    def test_create_failed(self):
        error_tag = '0:1.self_attention.core_attention_flash_0/rank0/output'
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2."
        step = 2
        with self.assertRaises(Exception) as context:
            self.AnomalyDataFactory.create(error_tag, message, step)
        self.assertEqual(str(context.exception), "tag must be a tuple with length 2")


class TestGradAnomalyData(TestCase):

    def setUp(self) -> None:
        tag_name = "0:1.self_attention.core_attention_flash.output:0/rank0/actv"
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash.output:0/rank0/actv', 'min') at step 2."
        group_mates = [0]
        self.GradAnomalyData = GradAnomalyData(tag_name=tag_name, message=message, group_mates=group_mates)

    def test_get_train_stage(self):
        tag_name_list = ["0:fc2.input:0/rank0/actv", "0:fc1.weight/rank0/post_grad", "0:fc2.weight/rank0/exp_avg_sq", ""]
        expected_train_stage_list = [0, 1, 2, -1]
        for tag_name, expected_train_stage in zip(tag_name_list, expected_train_stage_list):
            train_stage = GradAnomalyData.get_train_stage(tag_name)
            self.assertEqual(train_stage, expected_train_stage)

    def test_to_dict(self):
        expected = {
            'rank': 0,
            'step': 0,
            'micro_step': 0,
            'pp_stage': 0,
            'vpp_stage': 0,
            'call_id': 0,
            'tag_name': "0:1.self_attention.core_attention_flash.output:0/rank0/actv",
            'message': "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash.output:0/rank0/actv', 'min') at step 2.",
            'group_mates': [0]
        }

        self.assertEqual(self.GradAnomalyData.to_dict(), expected)

    def test_get_key(self):
        expected = "0:1.self_attention.core_attention_flash.output:0/rank0/actv_step_0_call_0"

        self.assertEqual(self.GradAnomalyData.get_key(), expected)

    def test_lt_different_step(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=2, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_different_micro_step(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=1, micro_step=1, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_same_micro_step_different_vpp_stage(self):
        # same forward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/actv")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=1, pp_stage=0, call_id=0, tag_name="xxx/actv")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

        # same backward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=1, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data2, data1)
        self.assertGreater(data1, data2)

        # diff train stage
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/actv")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=1, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_same_micro_step_same_vpp_stage_different_pp_stage(self):
        # same forward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/actv")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=1, call_id=0, tag_name="xxx/actv")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

        # same backward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=1, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data2, data1)
        self.assertGreater(data1, data2)

        # diff train stage
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/input")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=1, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_same_micro_step_same_vpp_stage_same_pp_stage_different_call_id(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=1, tag_name="")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_data(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        self.assertGreaterEqual(data1, data2)
        self.assertLessEqual(data1, data2)

    def test_lt_not_instance(self):
        data = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0)
        not_instance = "not an instance of GradAnomalyData"
        self.assertEqual(data.__lt__(not_instance), NotImplemented)

    def test_le_same_instance(self):
        # 测试相同实例的情况
        data1 = GradAnomalyData()
        self.assertTrue(data1 <= data1)

    def test_le_different_instance(self):
        # 测试不同实例的情况
        data1 = GradAnomalyData()
        data2 = GradAnomalyData()
        self.assertTrue(data1 <= data2)

    def test_le_not_instance(self):
        # 测试非GradAnomalyData实例的情况
        data = GradAnomalyData()
        not_instance = "Not an instance of GradAnomalyData"
        self.assertEqual(data.__le__(not_instance), NotImplemented)

    def test_le_different_instance_not_equal(self):
        # 测试不同实例且不相等的情况
        data1 = GradAnomalyData()
        data2 = GradAnomalyData()
        data2.some_attribute = "some value"
        self.assertTrue(data1 <= data2)


class TestAnomalyDataWriter(TestCase):

    def test_get_anomaly_dict(self):
        # 测试 get_anomaly_dict 方法
        anomaly1 = MagicMock()
        anomaly1.get_key.return_value = 'anomaly1'
        anomaly1.to_dict.return_value = {'value': 1}

        anomaly2 = MagicMock()
        anomaly2.get_key.return_value = 'anomaly2'
        anomaly2.to_dict.return_value = {'value': 2}

        anomalies = [anomaly1, anomaly2]
        result = AnomalyDataWriter.get_anomaly_dict(anomalies)

        expected = {
            'anomaly1': {'value': 1},
            'anomaly2': {'value': 2}
        }
        self.assertEqual(result, expected)

    @patch('msprobe.core.monitor.anomaly_processor.os.path.exists')
    @patch('msprobe.core.monitor.anomaly_processor.create_directory')
    @patch('msprobe.core.monitor.anomaly_processor.save_json')
    def test_init_detected_json(self, mock_save_json, mock_create_directory, mock_exists):
        # 模拟路径检查
        mock_exists.side_effect = [False, False, False]  # dump_path, dump_rank_dir, json_path
        # 模拟文件不存在
        mock_create_directory.side_effect = None

        writer = AnomalyDataWriter('/tmp/dump', 0)
        writer.init_detected_json()

        # 检查是否创建了目录
        mock_create_directory.assert_any_call('/tmp/dump/rank0')

        # 检查是否初始化了 JSON 文件
        mock_save_json.assert_called_once_with(writer.json_path, {}, indent=1)

    @patch('msprobe.core.monitor.anomaly_processor.check_file_or_directory_path')
    @patch('msprobe.core.monitor.anomaly_processor.remove_path')
    @patch('msprobe.core.monitor.anomaly_processor.save_json')
    @patch('msprobe.core.monitor.anomaly_processor.logger')
    def test_init_detected_json_existing_file(self, mock_logger, mock_save_json, mock_remove_path, mock_check_path):
        # 设置测试参数
        dump_path = 'test/dump_path'
        rank = 0
        writer = AnomalyDataWriter(dump_path, rank)

        # 模拟文件存在情况
        mock_check_path.side_effect = None  # 阻止实际调用
        mock_remove_path.return_value = None  # 阻止实际调用

        # 模拟 json_path 存在
        writer.json_path = 'existing_file.json'
        with patch('os.path.exists', return_value=True):
            writer.init_detected_json()

        # 验证文件删除和新文件保存
        mock_remove_path.assert_called_once_with(writer.json_path)
        mock_logger.warning.assert_called_once_with(f"The existing file will be deleted: {writer.json_path}.")
        mock_save_json.assert_called_once_with(writer.json_path, {}, indent=1)

    @patch('msprobe.core.monitor.anomaly_processor.os.path.exists')
    @patch('msprobe.core.monitor.anomaly_processor.load_json')
    @patch('msprobe.core.monitor.anomaly_processor.save_json')
    def test_write_detected_json(self, mock_save_json, mock_load_json, mock_exists):
        mock_exists.side_effect = [True, True]  # json_path 存在

        # 创建模拟的异常数据
        anomalies = [MagicMock(), MagicMock()]
        anomalies[0].get_key.return_value = 'anomaly1'
        anomalies[0].to_dict.return_value = {'value': 1}
        anomalies[1].get_key.return_value = 'anomaly2'
        anomalies[1].to_dict.return_value = {'value': 2}

        mock_load_json.return_value = {'existing_anomaly': {'value': 0}}

        writer = AnomalyDataWriter('/tmp/dump', 0)
        writer.write_detected_json(anomalies)

        expected_data = {
            'existing_anomaly': {'value': 0},
            'anomaly1': {'value': 1},
            'anomaly2': {'value': 2}
        }

        # 检查 JSON 是否被加载和保存
        mock_load_json.assert_called_once_with(writer.json_path)
        mock_save_json.assert_called_once_with(writer.json_path, expected_data, indent=1)


class TestAnomalyDataLoader(TestCase):

    @patch('msprobe.core.monitor.anomaly_processor.GradAnomalyData')  # 替换为 GradAnomalyData 的实际导入路径
    def test_create_instances_from_dict(self, mock_GradAnomalyData):
        # 模拟 GradAnomalyData 的构造函数
        def mock_constructor(**kwargs):
            return None

        mock_GradAnomalyData.side_effect = mock_constructor  # 假设构造成功

        data = {
            'anomaly1': {'key1': 'value1', 'key2': 'value2'},
            'anomaly2': {'key1': 'value3', 'key2': 'value4'},
        }

        loader = AnomalyDataLoader('/tmp/data')
        instances = loader.create_instances_from_dict(data)

        # 确保创建了两个实例，第三个因缺少 key2 被捕获
        self.assertEqual(len(instances), 2)

    @patch('msprobe.core.monitor.anomaly_processor.os.listdir')
    @patch('msprobe.core.monitor.anomaly_processor.os.path.exists')
    @patch('msprobe.core.monitor.anomaly_processor.load_json')
    @patch('msprobe.core.monitor.anomaly_processor.check_file_or_directory_path')
    @patch('msprobe.core.monitor.anomaly_processor.GradAnomalyData')
    def test_get_anomalies_from_jsons(self, mock_GradAnomalyData, mock_check_path, mock_load_json, mock_exists,
                                      mock_listdir):
        mock_check_path.return_value = None
        mock_listdir.return_value = ['rank0', 'rank1']

        # 模拟 rank0/anomaly.json 存在，rank1/anomaly.json 不存在
        mock_exists.side_effect = [True, False]
        mock_load_json.return_value = {
            'anomaly1': {'key1': 'value1', 'key2': 'value2'},
            'anomaly2': {'key1': 'value3', 'key2': 'value4'}
        }

        # 模拟 GradAnomalyData 的构造函数
        def mock_constructor(**kwargs):
            return None

        mock_GradAnomalyData.side_effect = mock_constructor  # 假设构造成功

        loader = AnomalyDataLoader('/tmp/data')
        with patch('msprobe.core.monitor.anomaly_processor.os.path.isdir', return_value=True):
            anomalies = loader.get_anomalies_from_jsons()

        # 确保从 rank0 读取了异常数据
        self.assertEqual(len(anomalies), 2)
        mock_check_path.assert_called_once_with('/tmp/data', isdir=True)
        mock_load_json.assert_called_once_with('/tmp/data/rank0/anomaly.json')


class TestAnomalyAnalyse(TestCase):

    def setUp(self):
        self.anomaly_analyse = AnomalyAnalyse()
        self.anomalies = [
            MagicMock(step=1, value=5),
            MagicMock(step=2, value=3),
            MagicMock(step=3, value=8),
            MagicMock(step=4, value=1),
        ]

    def test_get_range_top_k(self):
        anomalies = [
            GradAnomalyData(step=1, micro_step=1, vpp_stage=0, pp_stage=0, call_id=0, tag_name=""),
            GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name=""),
            GradAnomalyData(step=2, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name=""),
            GradAnomalyData(step=3, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        ]

        # step_list not empty
        result = self.anomaly_analyse.get_range_top_k(3, [1], anomalies)
        self.assertEqual(len(result), 2)
        result = self.anomaly_analyse.get_range_top_k(3, [1, 2], anomalies)
        self.assertEqual(len(result), 3)

        # top_k greater than anomalies length
        result = self.anomaly_analyse.get_range_top_k(4, [], anomalies)
        self.assertEqual(len(result), 4)

        # top_k less than anomalies length
        result = self.anomaly_analyse.get_range_top_k(3, [], anomalies)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, [anomalies[1], anomalies[0], anomalies[2]])

    @patch('msprobe.core.monitor.anomaly_processor.os.path.exists')
    @patch('msprobe.core.monitor.anomaly_processor.AnomalyDataWriter.get_anomaly_dict')
    @patch('msprobe.core.monitor.anomaly_processor.save_json')
    @patch('msprobe.core.monitor.anomaly_processor.logger')
    def test_rewrite_sorted_anomalies(self, mock_logger, mock_save_json, mock_get_anomaly_dict, mock_exists):
        # 设置 mock
        mock_exists.return_value = False
        mock_get_anomaly_dict.return_value = {'anomalies': 'data'}

        output_path = 'output_path'

        # 调用方法
        self.anomaly_analyse.sorted_anomalies = self.anomalies
        with patch("msprobe.core.monitor.anomaly_processor.check_file_or_directory_path", return_value=None):
            self.anomaly_analyse.rewrite_sorted_anomalies(output_path)

        # 验证调用
        mock_get_anomaly_dict.assert_called_once_with(self.anomaly_analyse.sorted_anomalies)
        mock_save_json.assert_called_once_with(
            os.path.join(output_path, 'anomaly_analyse.json'),
            {'anomalies': 'data'},
            indent=1
        )
        mock_logger.info.assert_called_once_with("anomaly_analyse.json is at output_path.")

    @patch('msprobe.core.monitor.anomaly_processor.os.path.exists')
    @patch('msprobe.core.monitor.anomaly_processor.logger')
    def test_rewrite_sorted_anomalies_file_exists(self, mock_logger, mock_exists):
        # 模拟文件已经存在的情况
        mock_exists.return_value = True
        output_path = 'output_path'

        # 调用方法
        with patch("msprobe.core.monitor.anomaly_processor.check_file_or_directory_path", return_value=None), \
                patch("msprobe.core.monitor.anomaly_processor.remove_path", return_value=None), \
                patch("msprobe.core.monitor.anomaly_processor.save_json", return_value=None):
            self.anomaly_analyse.rewrite_sorted_anomalies(output_path)

        # 验证日志警告
        mock_logger.warning.assert_called_once_with(
            f"The existing file will be deleted: output_path/anomaly_analyse.json.")


class TestGetStepAndStop(TestCase):

    def test_valid_step_list_and_top_k(self):
        # 构造有效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3]'
        args.top_k_number = 5

        step_list, top_k = _get_step_and_stop(args)

        self.assertEqual(step_list, [1, 2, 3])
        self.assertEqual(top_k, 5)

    def test_invalid_step_list(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3'  # 不完整的列表
        args.top_k_number = 5

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The step list must be a resolvable list type.")

    def test_non_list_step_list(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = 'not_a_list'  # 非列表
        args.top_k_number = 5

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The step list must be a resolvable list type.")

    def test_top_k_number_zero(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3]'
        args.top_k_number = 0  # 非法值

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The top k number must be greater than 0.")

    def test_top_k_number_negative(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3]'
        args.top_k_number = -1  # 非法值

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The top k number must be greater than 0.")


class TestAnomalyAnalyseFunction(TestCase):

    @patch('msprobe.core.monitor.anomaly_processor._get_parse_args')  # 模拟命令行参数解析
    @patch('msprobe.core.monitor.anomaly_processor._get_step_and_stop')  # 模拟步骤和顶级数字解析
    @patch('msprobe.core.monitor.anomaly_processor.AnomalyDataLoader')  # 模拟数据加载器
    @patch('msprobe.core.monitor.anomaly_processor.AnomalyAnalyse')  # 模拟异常分析器
    @patch('msprobe.core.monitor.anomaly_processor.logger')  # 模拟日志记录
    def test_anomaly_analyse(self, mock_logger, mock_anomaly_analyse, mock_anomaly_data_loader, mock_get_step_and_stop,
                             mock_get_parse_args):
        # 模拟命令行参数
        mock_args = MagicMock()
        mock_args.data_path_dir = 'path/to/data'
        mock_args.out_path = 'path/to/output'
        mock_args.step_list = '[1, 2, 3]'
        mock_args.top_k_number = 5
        mock_get_parse_args.return_value = mock_args

        # 模拟步骤和顶级数字
        mock_step_list = [1, 2, 3]
        mock_top_k_number = 5
        mock_get_step_and_stop.return_value = (mock_step_list, mock_top_k_number)

        # 模拟数据加载
        mock_loader_instance = MagicMock()
        mock_loader_instance.get_anomalies_from_jsons.return_value = [
            MagicMock(message='Anomaly 1'),
            MagicMock(message='Anomaly 2'),
            MagicMock(message='Anomaly 3')
        ]
        mock_anomaly_data_loader.return_value = mock_loader_instance

        # 模拟异常分析
        mock_analyser_instance = MagicMock()
        mock_analyser_instance.get_range_top_k.return_value = [
            MagicMock(message='Top Anomaly 1'),
            MagicMock(message='Top Anomaly 2')
        ]
        mock_anomaly_analyse.return_value = mock_analyser_instance

        # 调用被测试的函数
        _anomaly_analyse()

        # 验证调用
        mock_get_parse_args.assert_called_once()
        mock_get_step_and_stop.assert_called_once_with(mock_args)
        mock_anomaly_data_loader.assert_called_once_with(mock_args.data_path_dir)
        mock_loader_instance.get_anomalies_from_jsons.assert_called_once()
        mock_analyser_instance.get_range_top_k.assert_called_once_with(
            mock_top_k_number, mock_step_list, mock_loader_instance.get_anomalies_from_jsons.return_value
        )
        mock_analyser_instance.rewrite_sorted_anomalies.assert_called_once_with(mock_args.out_path)

        # 验证日志记录
        mock_logger.info.assert_any_call(f"Top {mock_top_k_number} anomalies are listed as follows:")
        mock_logger.info.assert_any_call("0: Top Anomaly 1")
        mock_logger.info.assert_any_call("1: Top Anomaly 2")


class TestParseArgs(TestCase):

    @patch('msprobe.core.monitor.anomaly_processor.sys.argv',
           new=['script_name', '-d', 'path/to/data', '-o', 'path/to/output', '-k', '5', '-s', '[1,2,3]'])
    def test_parse_args_with_all_arguments(self):
        args = _get_parse_args()
        self.assertEqual(args.data_path_dir, 'path/to/data')
        self.assertEqual(args.out_path, 'path/to/output')
        self.assertEqual(args.top_k_number, 5)
        self.assertEqual(args.step_list, '[1,2,3]')

    @patch('msprobe.core.monitor.anomaly_processor.sys.argv', new=['script_name', '-d', 'path/to/data'])
    def test_parse_args_with_required_argument_only(self):
        args = _get_parse_args()
        self.assertEqual(args.data_path_dir, 'path/to/data')
        self.assertEqual(args.out_path, '')
        self.assertEqual(args.top_k_number, 8)  # 默认值
        self.assertEqual(args.step_list, '[]')  # 默认值

    @patch('msprobe.core.monitor.anomaly_processor.sys.argv', new=['script_name', '-d', 'path/to/data', '-k', '10'])
    def test_parse_args_with_topk_only(self):
        args = _get_parse_args()
        self.assertEqual(args.data_path_dir, 'path/to/data')
        self.assertEqual(args.out_path, '')
        self.assertEqual(args.top_k_number, 10)  # 提供的值
        self.assertEqual(args.step_list, '[]')  # 默认值


if __name__ == '__main__':
    unittest.main()
