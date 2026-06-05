import unittest
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch
import sys

from msprobe.core.dump.dump2db.dump2db import (
    DumpRecordBuilder,
    TensorProcessingParams,
    scan_files,
    extract_root_nodes,
    reindex_keys_with_mapping,
)
from msprobe.core.dump.dump2db.db_utils import DumpDB
from msprobe.core.common.const import Const, Data2DBConst


class TestDumpRecordBuilderStaticMethods(unittest.TestCase):
    """测试DumpRecordBuilder的静态方法"""

    def test_extract_target_info_basic(self):
        """测试提取目标信息 - 基础情况"""
        full_key = "Module._fsdp_wrapped_module.blocks.18._fsdp_wrapper_module.ffn.1.GELU.18"
        target_prefix, vpp_stage, micro_step = DumpRecordBuilder.extract_target_info(
            full_key)
        self.assertEqual(
            target_prefix, "Module._fsdp_wrapped_module.blocks.18._fsdp_wrapper_module.ffn.1.GELU")
        self.assertEqual(vpp_stage, 0)
        self.assertEqual(micro_step, 18)

    def test_parse_tensor_target_batch(self):
        """批量测试解析tensor目标"""
        test_cases = [
            # (metric_type, tensor_type, tensor_idx, expected_result, description)
            (Data2DBConst.FORWARD, Const.INPUT_ARGS,
             0, ".input.0", "forward input type"),
            (Data2DBConst.FORWARD, Const.OUTPUT, 1,
             ".output.1", "forward output type"),
            (Data2DBConst.FORWARD, Const.PARAMS, "weight",
             ".parameters.weight", "forward parameters type"),
            (Data2DBConst.BACKWARD, Const.INPUT,
             0, ".input.0", "backward input type"),
            (Data2DBConst.BACKWARD, Const.OUTPUT, 1,
             ".output.1", "backward output type"),
            ("unknown_type", "unknown_tensor", 0, "", "unknown metric type"),
            (Data2DBConst.RECOMPUTE, Const.INPUT_ARGS,
             0, ".input.0", "recompute input type"),
            (Data2DBConst.RECOMPUTE, Const.OUTPUT, 1,
             ".output.1", "recompute output type")
        ]

        for metric_type, tensor_type, tensor_idx, expected, description in test_cases:
            with self.subTest(metric_type=metric_type, tensor_type=tensor_type,
                              tensor_idx=tensor_idx, description=description):
                result = DumpRecordBuilder.parse_tensor_target(
                    metric_type, tensor_type, tensor_idx)
                self.assertEqual(result, expected,
                                 f"Failed for {description}: {metric_type}, {tensor_type}, {tensor_idx}")


class TestDetermineMetricType(unittest.TestCase):
    """测试DumpRecordBuilder类determine_metric_type方法"""

    def setUp(self):
        """设置测试环境"""
        self.mock_db = Mock(spec=DumpDB)
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.cache_targets.return_value = {"id": 123}

        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir="/test/data",
            mapping={},
            micro_step=None
        )

    def test_determine_metric_type_batch(self):
        """批量测试确定metric类型"""
        test_cases = [
            # (full_key, tensor_data, expected_metric_type, expected_processed_key, description)
            ("model.layer1.forward.0", {"is_recompute": False},
             Data2DBConst.FORWARD, "model.layer1.0", "forward type with micro step"),
            ("model.layer1.forward.1", {"is_recompute": True},
             Data2DBConst.RECOMPUTE, "model.layer1.1", "recompute type with micro step"),
            ("model.layer1.backward.0", {},
             Data2DBConst.BACKWARD, "model.layer1.0", "backward type with micro step"),
            ("model.layer1.parameters_grad", {},
             Data2DBConst.PARAMETERS_GRAD, "model.layer1", "parameters_grad type"),
            ("module.forward_image.backward.0", {"is_recompute": True},
             Data2DBConst.BACKWARD, "module.forward_image.0", "backward with forward in name"),
            ("a.b.c.0.forward", {"is_recompute": False},
             Data2DBConst.FORWARD, "a.b.c.0", "operator forward style"),
            ("model.layer1", {"is_recompute": False},
             None, "model.layer1", "no metric type keyword"),
        ]

        for full_key, tensor_data, expected_metric_type, expected_processed_key, description in test_cases:
            with self.subTest(description=description, full_key=full_key):
                metric_type, processed_key = self.builder._determine_metric_type(
                    full_key, tensor_data)
                self.assertEqual(metric_type, expected_metric_type,
                                 f"Metric type mismatch for {description}")
                self.assertEqual(processed_key, expected_processed_key,
                                 f"Processed key mismatch for {description}")

    def test_determine_metric_type_with_mapping(self):
        """测试使用映射确定metric类型"""
        builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir="/test/data",
            mapping={"old_prefix": "new_prefix"},
            micro_step=None
        )
        full_key = "old_prefix.layer1.forward"
        tensor_data = {}
        metric_type, processed_key = builder._determine_metric_type(
            full_key, tensor_data)
        self.assertEqual(processed_key, "new_prefix.layer1")
        self.assertEqual(metric_type, Data2DBConst.FORWARD)


class TestTensorDataProcessing(unittest.TestCase):
    """Test class for tensor data processing methods"""

    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock(spec=DumpDB)
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.cache_targets.return_value = {"id": 123}

        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir="/test/data",
            mapping={},
            micro_step=None
        )

        # Common test data
        self.valid_tensor = {
            "type": "torch.Tensor",
            "dtype": "Float32",
            "Max": 1.0,
            "Min": 0.0
        }
        self.unsupported_type = {
            "type": "DTensor",
            "dtype": "Float32",
            "Max": 1.0,
            "Min": 0.0
        }
        self.unsupported_dtype = {
            "type": "torch.Tensor",
            "dtype": "Float4",
            "Max": 1.0,
            "Min": 0.0
        }

        self.common_tensor_params = {
            "target_prefix": "model.layer1",
            "vpp_stage": 0,
            "micro_step": 0,
            "step": 100,
            "rank": 0,
            "metric_id": 1
        }

    def _create_tensor_params(self, tensor_data, metric_type):
        """Helper method to create TensorProcessingParams"""
        return TensorProcessingParams(
            tensor_data=tensor_data,
            metric_type=metric_type,
            **self.common_tensor_params
        )

    def test_add_tensor_data(self):
        """Test adding tensor data"""
        target_name = "model.layer1.input.0"
        tensor_params = self._create_tensor_params({}, Data2DBConst.FORWARD)
        batch_data = []

        self.builder._add_tensor_data(
            self.valid_tensor, target_name, tensor_params, batch_data)

        # Verify cache targets was called
        self.mock_db.cache_targets.assert_called_once_with(
            (target_name, 0, 0), 1)

        # Verify batch data was correctly added
        self.assertEqual(len(batch_data), 1)
        row_data = batch_data[0]
        self.assertEqual(row_data[0], 0)  # rank
        self.assertEqual(row_data[1], 100)  # step
        self.assertEqual(row_data[2], {"id": 123})  # target cache
        self.assertEqual(row_data[3], 1)  # metric_id

    def test_process_forward_data_normal(self):
        """Test processing forward data with normal tensors"""
        tensor_data = {
            Const.INPUT_ARGS: [self.valid_tensor],
            Const.OUTPUT: [self.valid_tensor],
            Const.PARAMS: {
                "weight": [self.valid_tensor],
                "bias": [self.valid_tensor]
            }
        }

        tensor_params = self._create_tensor_params(
            tensor_data, Data2DBConst.FORWARD)
        batch_data = []

        self.builder._process_forward_data(tensor_params, batch_data)

        # 4个tensor加入batch (input_args and output, params)
        self.assertEqual(len(batch_data), 4)

    def test_process_backward_data_normal(self):
        """Test processing backward data with normal tensors"""
        tensor_data = {
            Const.INPUT: [self.valid_tensor],
            Const.OUTPUT: [self.valid_tensor]
        }

        tensor_params = self._create_tensor_params(
            tensor_data, Data2DBConst.BACKWARD)
        batch_data = []

        self.builder._process_backward_data(tensor_params, batch_data)

        # Should add two tensors (input and output)
        self.assertEqual(len(batch_data), 2)

    def test_process_parameters_data_normal(self):
        """Test processing parameters data with normal parameters"""
        tensor_data = {
            "weight": [self.valid_tensor],
            "bias": [self.valid_tensor]
        }

        tensor_params = self._create_tensor_params(
            tensor_data, Data2DBConst.PARAMETERS_GRAD)
        batch_data = []

        self.builder._process_parameters_data(tensor_params, batch_data)

        # 两个权重加入batch
        self.assertEqual(len(batch_data), 2)

    def test_process_recompute_data(self):
        """Test processing recompute data"""
        tensor_data = {
            Const.INPUT_ARGS: [self.valid_tensor],
            Const.OUTPUT: [self.valid_tensor]
        }

        tensor_params = self._create_tensor_params(
            tensor_data, Data2DBConst.RECOMPUTE)
        batch_data = []

        self.builder._process_forward_data(tensor_params, batch_data)
        # 两个张量信息
        self.assertEqual(len(batch_data), 2)

    def test_process_unsupported_data_with_different_metric_types(self):
        """Test adding tensor data with different metric types"""
        tensor_data = {
            Const.INPUT_ARGS: [self.valid_tensor, self.unsupported_type, self.unsupported_dtype],
            Const.OUTPUT: [self.valid_tensor, self.unsupported_type, self.unsupported_dtype],
            Const.INPUT: [self.valid_tensor, self.unsupported_type, self.unsupported_dtype],
            Const.PARAMS: {
                "weight": [self.valid_tensor, self.unsupported_type, self.unsupported_dtype],
                "bias": [self.valid_tensor, self.unsupported_type, self.unsupported_dtype]
            }
        }
        test_cases = [
            (self.builder._process_forward_data, "unsupported forward metric", 4),
            (self.builder._process_backward_data,
             "unsupported backward metric", 2),
            (self.builder._process_forward_data,
             "unsupported recompute metric", 4),
            (self.builder._process_parameters_data,
             "unsupported parameters_grad metric", 3),
        ]

        for process_func, description, in_batch_num in test_cases:
            with self.subTest(description=description):
                tensor_params = self._create_tensor_params(
                    tensor_data, "test_metric_type")
                batch_data = []
                process_func(tensor_params, batch_data)
                self.assertEqual(len(batch_data), in_batch_num)


class TestIntegration(unittest.TestCase):
    """DumpRecordBuilder核心方法集成测试类"""

    def setUp(self):
        """测试前置设置"""
        self.mock_db = Mock(spec=DumpDB)
        self.data_dir = tempfile.mkdtemp()
        self.mapping = {}
        self.micro_step = None

        # 创建builder实例
        self.builder = DumpRecordBuilder(
            self.mock_db, self.data_dir, self.mapping, self.micro_step
        )

        # 模拟数据库方法
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.cache_targets.return_value = {"id": 1}
        self.mock_db.batch_insert_targets.return_value = None
        self.mock_db.batch_insert_data.return_value = None
        self.mock_db.init_global_stats_data.return_value = None
        self.mock_db.extract_tags_from_processed_targets.return_value = None

    def tearDown(self):
        """测试后置清理"""
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

    def test_process_dump_file_integration(self):
        """测试_process_dump_file方法完整流程"""
        # 准备测试数据
        test_data = {
            'data': {
                'module.conv1.forward': {
                    Const.INPUT_ARGS: [
                        {
                            'type': 'torch.Tensor',
                            'dtype': 'Float16',
                            'Max': 1.0,
                            'Min': 0.0,
                            'Mean': 0.5,
                        }
                    ],
                    Const.OUTPUT: [
                        {
                            'type': 'torch.Tensor',
                            'dtype': 'Float16',
                            'Max': 2.0,
                            'Min': -1.0,
                            'Mean': 0.8,
                        }
                    ]
                },
                'module.fc.backward': {
                    Const.INPUT: [
                        {
                            'type': 'torch.Tensor',
                            'dtype': 'Float32',
                            'Max': 3.0,
                            'Min': -2.0,
                            'Mean': 0.3,
                        }
                    ]
                }
            }
        }

        # 创建临时dump文件
        dump_file_path = os.path.join(self.data_dir, 'dump.json')
        with open(dump_file_path, 'w') as f:
            json.dump(test_data, f)

        # 模拟get_metric_id根据不同类型返回不同ID
        def mock_get_metric_id(metric_type):
            metric_map = {
                'forward': 1,
                'backward': 2,
                'recompute': 3,
                'parameters_grad': 4
            }
            return metric_map.get(metric_type, 1)

        self.mock_db.get_metric_id.side_effect = mock_get_metric_id

        # 执行测试
        self.builder._process_dump_file(
            dump_file_path, "", 0, 2)

        self.mock_db.batch_insert_data.assert_called()
        self.mock_db.batch_insert_targets.assert_called()

        # 验证插入的数据结构
        batch_data_call = self.mock_db.batch_insert_data.call_args[0][0]
        self.assertIsInstance(batch_data_call, list)
        self.assertEqual(len(batch_data_call), 3)

    def test_import_data_full_integration(self):
        """测试import_data方法完整流程"""
        # 创建模拟的目录结构
        step_dirs = ['step0', 'step1']
        rank_dirs = ['rank0', 'rank1']

        # 创建目录和dump文件
        for step_dir in step_dirs:
            step_path = os.path.join(self.data_dir, step_dir)
            os.makedirs(step_path, exist_ok=True)

            for rank_dir in rank_dirs:
                rank_path = os.path.join(step_path, rank_dir)
                os.makedirs(rank_path, exist_ok=True)

                # 创建dump.json文件
                dump_data = {'data': {}}

                dump_file_path = os.path.join(rank_path, 'dump.json')
                with open(dump_file_path, 'w') as f:
                    json.dump(dump_data, f)

        # 模拟tqdm不显示进度条
        with patch('msprobe.core.dump.dump2db.dump2db.tqdm') as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x

            # 模拟get_metric_id
            self.mock_db.get_metric_id.return_value = 1
            # 模拟table_name_cache
            valid_ranks = scan_files(self.data_dir)
            with patch.object(self.builder, '_process_dump_file') as mock_process_dump:
                # 执行测试
                self.builder.import_data(valid_ranks)
                # 验证全局统计更新被调用
                self.mock_db.init_global_stats_data.assert_called_once()
                # 验证处理了正确数量的step
                self.assertEqual(mock_process_dump.call_count,
                                 len(step_dirs) * len(rank_dirs))


class TestExtractRootNodes(unittest.TestCase):
    """测试extract_root_nodes函数"""

    def test_extract_root_nodes_normal(self):
        """正常数据：根节点映射"""
        data = {
            "Module.layer1.forward": 3,
            "Module.layer2.forward": 3,
            "Module.layer1.backward": "Module.layer1.forward",
            "Module.layer2.backward": "Module.layer2.forward",
        }
        result = extract_root_nodes(data)
        self.assertEqual(result["Module.layer1.forward"], 3)
        self.assertEqual(result["Module.layer2.forward"], 3)
        self.assertIn("Module.layer1.backward", result)
        self.assertIn("Module.layer2.backward", result)

    def test_extract_root_nodes_with_none_values(self):
        """含None值的节点处理（backward标记的父节点为None）"""
        data = {
            "Module.layer1.forward": 0,
            "Module.layer2.forward": 1,
        }
        with patch("msprobe.core.dump.dump2db.dump2db.Const.SEP", "."), \
             patch("msprobe.core.dump.dump2db.dump2db.Const.BACKWARD", "backward"):
            result = extract_root_nodes(data)
        self.assertEqual(result["Module.layer1.forward"], 0)
        self.assertEqual(result["Module.layer2.forward"], 1)

    def test_extract_root_nodes_backward_parent_none(self):
        """backward对应的value为None时会替换为forward并查找根节点"""
        data = {
            "Module.layer1.forward": 0,
            "Module.layer1.backward": None,
        }
        root_nodes = extract_root_nodes(data)
        # backward会替换为forward，找到根节点0
        self.assertIn("Module.layer1.backward", root_nodes)
        self.assertEqual(root_nodes["Module.layer1.backward"], 0)


class TestReindexKeysWithMapping(unittest.TestCase):
    """测试reindex_keys_with_mapping函数"""

    def test_reindex_simple(self):
        """相同value的不同prefix各自从0重新编号"""
        original = {
            "Module.layer1.forward.10": 3,
            "Module.layer2.forward.11": 3,
        }
        result = reindex_keys_with_mapping(original)
        self.assertIn("Module.layer1.forward.0", result.values())
        # 不同prefix，各自从0开始编号
        self.assertIn("Module.layer2.forward.0", result.values())

    def test_reindex_with_different_groups(self):
        """不同value分组"""
        original = {
            "Module.layer1.forward.10": 0,
            "Module.layer2.forward.20": 1,
        }
        result = reindex_keys_with_mapping(original)
        self.assertIn("Module.layer1.forward.0", result.values())
        self.assertIn("Module.layer2.forward.0", result.values())

    def test_reindex_two_digit_suffix(self):
        """倒数第二位是数字的key"""
        original = {
            "Module.layer1.10.forward": 3,
        }
        # parts[-1]="forward"非数字，parts[-2]="10"是数字 -> 重写倒数第二位
        result = reindex_keys_with_mapping(original)
        self.assertIn("Module.layer1.0.forward", result.values())

    def test_reindex_no_digit_key(self):
        """没有数字后缀的key"""
        original = {
            "Module.layer.forward": 3,
        }
        # 最后两位都不是数字，用整个key作为前缀
        result = reindex_keys_with_mapping(original)
        self.assertIn("Module.layer.forward", result.values())


class TestScanFilesAdditional(unittest.TestCase):
    """测试scan_files补充场景"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dump_file(self, step_dir, item_dir):
        """在指定step/item下创建dump.json和construct.json"""
        item_path = os.path.join(self.temp_dir, step_dir, item_dir)
        os.makedirs(item_path, exist_ok=True)
        with open(os.path.join(item_path, "dump.json"), "w") as f:
            json.dump({"data": {}}, f)
        with open(os.path.join(item_path, "construct.json"), "w") as f:
            json.dump({}, f)

    def test_scan_files_empty_dir(self):
        """空目录返回空结果"""
        result = scan_files(self.temp_dir)
        self.assertEqual(result, {})

    def test_scan_files_invalid_step_name(self):
        """无效step名称应跳过"""
        os.makedirs(os.path.join(self.temp_dir, "not_a_step"), exist_ok=True)
        result = scan_files(self.temp_dir)
        self.assertEqual(result, {})

    def test_scan_files_no_json_files(self):
        """有step目录但无dump.json时该step不会出现在结果中"""
        rank_path = os.path.join(self.temp_dir, "step0", "rank0")
        os.makedirs(rank_path, exist_ok=True)
        result = scan_files(self.temp_dir)
        self.assertNotIn(0, result)

    def test_scan_files_proc_directories(self):
        """proc目录正确映射为rank编号"""
        self._create_dump_file("step0", "rank0")
        self._create_dump_file("step0", "proc0")
        self._create_dump_file("step0", "proc1")
        result = scan_files(self.temp_dir)
        self.assertIn(0, result)
        self.assertEqual(len(result[0]), 3)  # rank0 + proc0 + proc1
        # proc0 -> rank1, proc1 -> rank2
        rank_ids = [rank for rank, _, _ in result[0]]
        self.assertIn(0, rank_ids)
        self.assertIn(1, rank_ids)
        self.assertIn(2, rank_ids)


class TestDetermineMetricTypeFSDP(unittest.TestCase):
    """测试_determine_metric_type的FSDP场景"""

    def setUp(self):
        self.mock_db = Mock(spec=DumpDB)
        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir="/test/data",
            mapping={},
            micro_step=None
        )

    def test_fsdp_parameters_grad(self):
        """FSDP格式的parameters_grad检测"""
        # fsdp格式：parts[-2] == Const.PARAMS_GRAD
        full_key = "Module.fsdp_unit.parameters_grad.something"
        with patch("msprobe.core.dump.dump2db.dump2db.Const.SEP", "."), \
             patch("msprobe.core.dump.dump2db.dump2db.Const.PARAMS_GRAD", "parameters_grad"):
            metric_type, processed_key = self.builder._determine_metric_type(full_key, {})
            self.assertEqual(metric_type, Data2DBConst.PARAMETERS_GRAD)
            self.assertEqual(processed_key, "Module.fsdp_unit")


class TestProcessForwardDataUnsupported(unittest.TestCase):
    """测试_process_forward_data中unsupported type/dtype的跳过逻辑"""

    def setUp(self):
        self.mock_db = Mock(spec=DumpDB)
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.cache_targets.return_value = {"id": 123}

        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir="/test/data",
            mapping={},
            micro_step=None
        )

        self.valid_tensor = {
            "type": "torch.Tensor",
            "dtype": "Float32",
            "Max": 1.0,
            "Min": 0.0
        }
        self.common_params = {
            "target_prefix": "model.layer1",
            "vpp_stage": 0,
            "micro_step": 0,
            "step": 100,
            "rank": 0,
            "metric_id": 1
        }

    def test_forward_data_unsupported_param_type(self):
        """parameters中非dict/非list类型跳过"""
        tensor_data = {
            Const.PARAMS: {
                "weight": "not_a_list",  # 非list类型
                "bias": [self.valid_tensor]
            }
        }
        tensor_params = TensorProcessingParams(
            tensor_data=tensor_data,
            metric_type=Data2DBConst.FORWARD,
            **self.common_params
        )
        batch_data = []
        self.builder._process_forward_data(tensor_params, batch_data)
        # weight跳过，bias正常处理
        self.assertEqual(len(batch_data), 1)

    def test_forward_data_unsupported_param_dtype(self):
        """parameters中不支持的dtype跳过"""
        unsupported_dtype = {
            "type": "torch.Tensor",
            "dtype": "Float4",
            "Max": 1.0,
            "Min": 0.0
        }
        tensor_data = {
            Const.PARAMS: {
                "weight": [unsupported_dtype],
                "bias": [self.valid_tensor]
            }
        }
        tensor_params = TensorProcessingParams(
            tensor_data=tensor_data,
            metric_type=Data2DBConst.FORWARD,
            **self.common_params
        )
        batch_data = []
        self.builder._process_forward_data(tensor_params, batch_data)
        # weight跳过，bias正常处理
        self.assertEqual(len(batch_data), 1)


class TestProcessParametersDataUnsupported(unittest.TestCase):
    """测试_process_parameters_data中unsupported dtype的跳过"""

    def setUp(self):
        self.mock_db = Mock(spec=DumpDB)
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.cache_targets.return_value = {"id": 123}

        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir="/test/data",
            mapping={},
            micro_step=None
        )

        self.common_params = {
            "target_prefix": "model.layer1",
            "vpp_stage": 0,
            "micro_step": 0,
            "step": 100,
            "rank": 0,
            "metric_id": 1
        }

    def test_parameters_data_unsupported_dtype(self):
        """不支持的dtype跳过"""
        unsupported_dtype = {
            "type": "torch.Tensor",
            "dtype": "Float4",
            "Max": 1.0,
            "Min": 0.0
        }
        valid_tensor = {
            "type": "torch.Tensor",
            "dtype": "Float32",
            "Max": 1.0,
            "Min": 0.0
        }
        tensor_data = {
            "weight": [unsupported_dtype],
            "bias": [valid_tensor]
        }
        tensor_params = TensorProcessingParams(
            tensor_data=tensor_data,
            metric_type=Data2DBConst.PARAMETERS_GRAD,
            **self.common_params
        )
        batch_data = []
        self.builder._process_parameters_data(tensor_params, batch_data)
        # weight跳过，bias正常处理
        self.assertEqual(len(batch_data), 1)


class TestProcessDumpFileEdgeCases(unittest.TestCase):
    """测试_process_dump_file的边界场景"""

    def setUp(self):
        self.mock_db = Mock(spec=DumpDB)
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.cache_targets.return_value = {"id": 123}
        self.data_dir = tempfile.mkdtemp()

        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir=self.data_dir,
            mapping={},
            micro_step=None
        )

    def tearDown(self):
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def _create_json_file(self, data):
        """创建临时JSON文件"""
        file_path = os.path.join(self.data_dir, "dump.json")
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    def test_process_dump_file_no_data_field(self):
        """缺少'data'字段返回None"""
        file_path = self._create_json_file({"other": "data"})
        result = self.builder._process_dump_file(file_path, "", 0, 0)
        self.assertIsNone(result)

    def test_process_dump_file_empty_data(self):
        """空的data字典"""
        file_path = self._create_json_file({"data": {}})
        result = self.builder._process_dump_file(file_path, "", 0, 0)
        self.assertIsNotNone(result)

    def test_process_dump_file_non_dict_tensor_data(self):
        """非dict类型的tensor_data跳过"""
        file_path = self._create_json_file({
            "data": {
                "module.conv1.forward": [1, 2, 3]  # 非dict类型
            }
        })
        result = self.builder._process_dump_file(file_path, "", 0, 0)
        self.assertIsNotNone(result)

    def test_process_dump_file_non_string_key(self):
        """非字符串类型的key跳过"""
        file_path = self._create_json_file({
            "data": {
                123: {"some": "data"}  # 非字符串key
            }
        })
        result = self.builder._process_dump_file(file_path, "", 0, 0)
        self.assertIsNotNone(result)


class TestProcessDumpFileMicroStep(unittest.TestCase):
    """测试micro_step模式下的_process_dump_file"""

    def setUp(self):
        self.mock_db = Mock(spec=DumpDB)
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.cache_targets.return_value = {"id": 123}
        self.data_dir = tempfile.mkdtemp()

        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir=self.data_dir,
            mapping={},
            micro_step=True
        )
        self.valid_tensor = {
            "type": "torch.Tensor",
            "dtype": "Float32",
            "Max": 1.0,
            "Min": 0.0
        }

    def tearDown(self):
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def _create_json_file(self, data):
        """创建临时JSON文件"""
        file_path = os.path.join(self.data_dir, "dump.json")
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    def _create_construct_file(self, data):
        """创建临时construct.json文件"""
        file_path = os.path.join(self.data_dir, "construct.json")
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    def test_process_with_construct_and_forward(self):
        """construct.json + forward数据的micro_step处理"""
        dump_path = self._create_json_file({
            "data": {
                "Module.layer1.conv.forward.0": {
                    Const.INPUT_ARGS: [self.valid_tensor]
                }
            }
        })
        construct_path = self._create_construct_file({
            "Module.layer1.conv.forward.0": 0
        })

        # mock extract_root_nodes 和 reindex_keys_with_mapping 简化测试
        with patch("msprobe.core.dump.dump2db.dump2db.extract_root_nodes",
                   return_value={"Module.layer1.conv.forward.0": 0}):
            with patch("msprobe.core.dump.dump2db.dump2db.reindex_keys_with_mapping",
                       return_value={"Module.layer1.conv.forward.0": "Module.layer1.conv.forward.0"}):
                result = self.builder._process_dump_file(dump_path, construct_path, 0, 0)
                self.assertIsNotNone(result)


class TestImportDataMicroStep(unittest.TestCase):
    """测试micro_step=True场景下的import_data"""

    def setUp(self):
        self.mock_db = Mock(spec=DumpDB)
        self.data_dir = tempfile.mkdtemp()

        # 创建step0/rank0 目录
        os.makedirs(os.path.join(self.data_dir, "step0", "rank0"), exist_ok=True)
        with open(os.path.join(self.data_dir, "step0", "rank0", "dump.json"), "w") as f:
            json.dump({"data": {}}, f)
        with open(os.path.join(self.data_dir, "step0", "rank0", "construct.json"), "w") as f:
            json.dump({}, f)

        self.builder = DumpRecordBuilder(
            db=self.mock_db,
            data_dir=self.data_dir,
            mapping={},
            micro_step=True
        )

    def tearDown(self):
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_import_data_with_micro_step(self):
        """micro_step=True时import_data正常执行"""
        self.mock_db.get_metric_id.return_value = 1
        self.mock_db.init_global_stats_data.return_value = None
        self.mock_db.extract_tags_from_processed_targets.return_value = None

        valid_ranks = scan_files(self.data_dir)
        with patch("msprobe.core.dump.dump2db.dump2db.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x
            with patch.object(self.builder, "_process_dump_file", return_value=1) as mock_process:
                self.builder.import_data(valid_ranks)
                self.mock_db.init_global_stats_data.assert_called_once()
                mock_process.assert_called_once()
