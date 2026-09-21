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

from msprobe.core.common.megatron_utils import (
    MegatronStepInfo,
    wrap_megatron_step,
    get_micro_step,
    is_megatron,
    VisualizationConfig,
    ModelParallelismVisualizer,
    ParallelConfig,
    ModelLayerSimulator,
    plot_model_parallelism
)
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


class TestMegatronStepInfo(unittest.TestCase):
    """测试 MegatronStepInfo 类"""

    def setUp(self):
        """每个测试前重置状态"""
        MegatronStepInfo.reset()
    
    def tearDown(self):
        """每个测试后清理状态"""
        MegatronStepInfo.reset()

    def test_initial_state(self):
        """测试初始状态"""
        self.assertFalse(MegatronStepInfo.is_megatron)
        self.assertFalse(MegatronStepInfo.is_forward)
        self.assertFalse(MegatronStepInfo.is_backward)
        self.assertEqual(MegatronStepInfo.forward_micro_step, -1)
        self.assertEqual(MegatronStepInfo.backward_micro_step, -1)

    def test_reset(self):
        """测试重置功能"""
        # 设置一些状态
        MegatronStepInfo.is_megatron = True
        MegatronStepInfo.is_forward = True
        MegatronStepInfo.forward_micro_step = 5

        MegatronStepInfo.reset()

        self.assertFalse(MegatronStepInfo.is_megatron)
        self.assertFalse(MegatronStepInfo.is_forward)
        self.assertEqual(MegatronStepInfo.forward_micro_step, -1)


class TestMegatronStepDecorators(unittest.TestCase):
    """测试 Megatron 步骤装饰器"""

    def setUp(self):
        MegatronStepInfo.reset()
    
    def tearDown(self):
        """每个测试后清理状态"""
        MegatronStepInfo.reset()

    def test_wrap_megatron_step_forward(self):
        """测试前向步骤装饰器"""

        @wrap_megatron_step
        def test_function():
            return "test_result"

        result = test_function()

        self.assertEqual(result, "test_result")
        self.assertTrue(MegatronStepInfo.is_megatron)
        self.assertTrue(MegatronStepInfo.is_forward)
        self.assertEqual(MegatronStepInfo.forward_micro_step, 0)

    def test_get_micro_step(self):
        """测试获取微步骤"""
        MegatronStepInfo.is_forward = True
        MegatronStepInfo.forward_micro_step = 3
        self.assertEqual(get_micro_step(), 3)

        MegatronStepInfo.is_forward = False
        MegatronStepInfo.is_backward = True
        MegatronStepInfo.backward_micro_step = 2
        self.assertEqual(get_micro_step(), 2)

    def test_is_megatron(self):
        """测试判断是否为 Megatron 模式"""
        self.assertFalse(is_megatron())
        MegatronStepInfo.is_megatron = True
        self.assertTrue(is_megatron())
        MegatronStepInfo.reset()


class TestVisualizationConfig(unittest.TestCase):
    """测试 VisualizationConfig 类"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = VisualizationConfig()

        self.assertEqual(config.height_per_tp, 0.5)
        self.assertEqual(config.width_per_stage, 2)
        self.assertEqual(config.min_width, 6)
        self.assertEqual(config.max_width, 60)
        self.assertEqual(config.font_sizes['title'], 14)
        self.assertEqual(config.font_sizes['text'], 9)

    def test_custom_initialization(self):
        """测试自定义初始化"""
        custom_font_sizes = {'title': 16, 'text': 10}
        config = VisualizationConfig(
            height_per_tp=0.8,
            width_per_stage=3,
            font_sizes=custom_font_sizes
        )

        self.assertEqual(config.height_per_tp, 0.8)
        self.assertEqual(config.width_per_stage, 3)
        self.assertEqual(config.font_sizes['title'], 16)
        self.assertEqual(config.font_sizes['text'], 10)


class TestParallelConfig(unittest.TestCase):
    """测试 ParallelConfig 类 - 详细异常场景测试"""

    def test_basic_initialization(self):
        """测试基础配置初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=8,
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                output_path=temp_dir
            )

        self.assertEqual(config.world_size, 8)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.pipeline_parallel_size, 2)
        self.assertEqual(config.data_parallel_size, 2)  # 8/(2 * 2)=2

    def test_derived_parameters_calculation(self):
        """测试派生参数计算"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=64,
                num_layers=96,
                tensor_parallel_size=4,
                pipeline_parallel_size=4,
                output_path=temp_dir
            )

        # 验证派生参数
        self.assertEqual(config.data_parallel_size, 4)  # 64/(4 * 4)=4
        self.assertEqual(config.transformer_pipeline_parallel_size, 4)

    def test_standalone_embedding_stage(self):
        """测试独立嵌入阶段配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=16,
                num_layers=48,
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
                standalone_embedding_stage=True,
                output_path=temp_dir
            )

        self.assertEqual(config.transformer_pipeline_parallel_size, 3)  # 4-1

    def test_virtual_pipeline_configuration(self):
        """测试虚拟流水线配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=32,
                num_layers=96,
                tensor_parallel_size=4,
                pipeline_parallel_size=4,
                num_layers_per_virtual_pipeline_stage=8,
                output_path=temp_dir
            )

        # 96/4=24 layers per stage, 24/8=3 virtual stages
        self.assertEqual(config.virtual_pipeline_parallel_size, 3)

    def test_output_path_creation(self):
        """测试输出路径创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=8,
                num_layers=24,
                output_path=temp_dir
            )

            self.assertTrue(config.output_path.endswith('.png'))
            self.assertTrue('ws8_ln24' in config.output_path)

    def test_invalid_world_size_scenarios(self):
        """测试无效world_size"""
        # 测试world_size为0
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(world_size=0, num_layers=24)
        self.assertIn("must be a positive integer", str(cm.exception))

        # 测试world_size超过最大值
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(world_size=1025, num_layers=24)
        self.assertIn(
            "exceeds the maximum supported value of 1024", str(cm.exception))

        # 测试world_size为浮点数（应该转换为整数失败）
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(world_size=8.5, num_layers=24)  # 浮点数
        self.assertIn("must be a positive integer", str(cm.exception))

    def test_invalid_num_layers_scenarios(self):
        """测试无效num_layers的各种场景"""
        # 测试num_layers为0
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(world_size=8, num_layers=0)
        self.assertIn("must be a positive integer", str(cm.exception))

        # 测试num_layers超过最大值
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(world_size=8, num_layers=257)
        self.assertIn(
            "exceeds the maximum supported value of 256", str(cm.exception))

    def test_standalone_embedding_divisibility(self):
        """测试独立嵌入阶段的整除性错误"""
        # 当启用独立嵌入阶段时，层数必须能被(pipeline_parallel_size-1)整除
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(
                world_size=8,
                num_layers=25,  # 25不能被(4-1)=3整除
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
                standalone_embedding_stage=True
            )
        self.assertIn("must be divisible by", str(cm.exception))
        self.assertIn("pipeline_parallel_size-1", str(cm.exception))

    def test_virtual_pipeline_configuration_errors(self):
        """测试虚拟流水线配置错误"""
        # 测试虚拟流水线但pipeline_parallel_size<=1
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(
                world_size=4,
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=1,  # PP=1不能使用虚拟流水线
                num_layers_per_virtual_pipeline_stage=6
            )
        self.assertIn(
            "pipeline-model-parallel size should be greater than 1", str(cm.exception))

        # 测试虚拟流水线层数不整除
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(
                world_size=8,
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
                num_layers_per_virtual_pipeline_stage=5  # 24/4=6, 6不能被5整除
            )
        self.assertIn("must be divisible by", str(cm.exception))

    def test_divisible_error_messages(self):
        """测试全面的整除性错误消息内容"""
        # 测试整除性错误消息
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(world_size=10, num_layers=24,
                           tensor_parallel_size=3, pipeline_parallel_size=3)
        error_msg = str(cm.exception)
        self.assertIn("world_size (10) is not divisible by", error_msg)
        self.assertIn(
            "tensor_parallel_size(3) x pipeline_parallel_size (3)", error_msg)

        # 测试层数整除性错误消息
        with self.assertRaises(ValueError) as cm:
            ParallelConfig(world_size=8, num_layers=25,
                           tensor_parallel_size=2, pipeline_parallel_size=2)
        error_msg = str(cm.exception)
        self.assertIn("must be divisible by", error_msg)
        self.assertIn("pipeline_parallel_size (2)", error_msg)


class TestModelLayerSimulator(unittest.TestCase):
    """测试 ModelLayerSimulator 类"""

    def setUp(self):
        """设置测试环境"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config = ParallelConfig(
                world_size=8,
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
                output_path=temp_dir
            )
        # 模拟PP阶段列表
        self.pp_stage_list = [
            [0, 1], [2, 3], [4, 5],[6, 7]
        ]

    def test_standard_pipeline_model(self):
        """测试标准流水线模型生成"""
        simulator = ModelLayerSimulator(self.config, self.pp_stage_list)

        # 测试阶段0（第一个阶段）
        model_stage0 = simulator.get_model_for_rank(0)
        self.assertEqual(len(model_stage0), 1)  # 单阶段配置
        self.assertIn("Embedding", model_stage0[0])  # 第一个阶段包含嵌入层

        # 测试阶段3（最后一个阶段）
        model_stage3 = simulator.get_model_for_rank(6)
        self.assertIn("OutputLayer", model_stage3[0])  # 最后一个阶段包含输出层

    def test_virtual_pipeline_model(self):
        """测试虚拟流水线模型生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=8,
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
                virtual_pipeline_parallel_size=2,
                output_path=temp_dir
            )

        simulator = ModelLayerSimulator(config, self.pp_stage_list)
        model = simulator.get_model_for_rank(0)

        # 虚拟流水线应该生成多个阶段配置
        self.assertEqual(len(model), 2)

    def test_transformer_layers_calculation(self):
        """测试Transformer层数计算"""
        simulator = ModelLayerSimulator(self.config, self.pp_stage_list)

        # 每个阶段应该处理 24/4 = 6 层
        model = simulator.get_model_for_rank(2)
        transformer_layers = [
            layer for layer in model[0] if isinstance(layer, int)]
        self.assertEqual(len(transformer_layers), 6)

    def test_standalone_embedding_stage(self):
        """测试独立嵌入阶段"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=8,
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
                standalone_embedding_stage=True,
                output_path=temp_dir
            )

        simulator = ModelLayerSimulator(config, self.pp_stage_list)

        # 嵌入阶段不应该包含Transformer层
        embedding_stage_model = simulator.get_model_for_rank(0)
        transformer_layers = [
            layer for layer in embedding_stage_model[0] if isinstance(layer, int)]
        self.assertEqual(len(transformer_layers), 0)

        # Transformer阶段应该包含层
        transformer_stage_model = simulator.get_model_for_rank(2)
        transformer_layers = [
            layer for layer in transformer_stage_model[0] if isinstance(layer, int)]
        self.assertTrue(len(transformer_layers) > 0)


class TestModelParallelismVisualizer(unittest.TestCase):
    """测试 ModelParallelismVisualizer 类 - 完善版"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试配置
        with tempfile.TemporaryDirectory() as temp_dir:
            self.args = ParallelConfig(
                world_size=16,
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
                data_parallel_size=1,
                output_path=temp_dir
            )

        # 模拟分组字典
        self.groups_dict = {
            "tp": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
            "dp": [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]],
            # 4个PP阶段
            "pp": [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
        }

        # 模拟rank模型配置
        self.rank_models = {
            0: [['Embedding', 1, 2, 3, 4, 5, 6]],
            1: [['Embedding', 1, 2, 3, 4, 5, 6]],
            2: [['Embedding', 1, 2, 3, 4, 5, 6]],
            3: [['Embedding', 1, 2, 3, 4, 5, 6]],
            4: [[7, 8, 9, 10, 11, 12]],
            5: [[7, 8, 9, 10, 11, 12]],
            6: [[7, 8, 9, 10, 11, 12]],
            7: [[7, 8, 9, 10, 11, 12]],
            8: [[13, 14, 15, 16, 17, 18]],
            9: [[13, 14, 15, 16, 17, 18]],
            10: [[13, 14, 15, 16, 17, 18]],
            11: [[13, 14, 15, 16, 17, 18]],
            12: [[19, 20, 21, 22, 23, 24, 'OutputLayer']],
            13: [[19, 20, 21, 22, 23, 24, 'OutputLayer']],
            14: [[19, 20, 21, 22, 23, 24, 'OutputLayer']],
            15: [[19, 20, 21, 22, 23, 24, 'OutputLayer']]
        }

        # 创建可视化器实例
        self.visualizer = ModelParallelismVisualizer(
            self.args, self.groups_dict, self.rank_models
        )

    def test_layout_calculation(self):
        """测试布局计算"""
        layout = self.visualizer.layout

        # 验证布局参数
        self.assertEqual(layout['num_layers'], 24)
        self.assertEqual(layout['num_tp'], 8)  # 8个TP组
        self.assertEqual(layout['num_model_copies'], 2)  # 2个DP组
        self.assertEqual(layout['tp_size'], 2)
        self.assertEqual(layout['pp_size'], 4)
        self.assertEqual(layout['vpp_size'], 1)

        # 验证模型代表
        self.assertIn((0, 0), layout['model_representatives'])
        self.assertIn((1, 1), layout['model_representatives'])
        self.assertEqual(len(layout['model_representatives']), 8)  # 2x4组合

    def test_figure_size_calculation(self):
        """测试图表尺寸计算"""
        fig_width, fig_height = self.visualizer._calculate_figure_size()

        # 验证尺寸在合理范围内
        self.assertGreater(fig_width, self.visualizer.config.min_width)
        self.assertLess(fig_width, self.visualizer.config.max_width)
        self.assertEqual(fig_height, self.visualizer.config.min_height)
        self.assertLess(fig_height, self.visualizer.config.max_height)

        # 验证包含图例宽度
        self.assertGreater(fig_width, self.visualizer.config.legend_width)

    def test_color_setup_small_scale(self):
        """测试小规模颜色配置"""
        # 测试小规模（<=20）
        self.visualizer.layout['num_model_copies'] = 8
        self.visualizer._setup_colors()

        self.assertEqual(len(self.visualizer.model_colors), 8)
        # 应该使用tab20色板
        self.assertTrue(hasattr(self.visualizer.model_colors, 'shape'))

    def test_color_setup_medium_scale(self):
        """测试中等规模颜色配置"""
        # 测试中等规模（21-40）
        self.visualizer.layout['num_model_copies'] = 25
        self.visualizer._setup_colors()

        self.assertEqual(len(self.visualizer.model_colors), 40)

    def test_color_setup_large_scale(self):
        """测试大规模颜色配置"""
        # 测试大规模（>40）
        self.visualizer.layout['num_model_copies'] = 50
        self.visualizer._setup_colors()

        self.assertEqual(len(self.visualizer.model_colors), 50)
        # 应该使用连续色板
        self.assertIsInstance(self.visualizer.model_colors, list)

    def test_layer_text_creation(self):
        """测试层文本创建"""
        test_cases = [
            # (输入层列表, 期望输出)
            (["Embedding", 1, 2, 3], "Embed+L1-3"),
            ([7, 8, 9, "OutputLayer"], "L7-9+Out"),
            ([4, 5, 6], "L4-6"),
            (["Embedding", 1, 2, 3, "OutputLayer"], "Embed+L1-3+Out"),
            ([], ""),  # 空列表
            (["Embedding"], "Embed"),  # 只有嵌入层
            (["OutputLayer"], "Out"),  # 只有输出层
        ]

        for layers, expected in test_cases:
            with self.subTest(layers=layers):
                result = self.visualizer._create_layer_text(layers)
                self.assertEqual(result, expected)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_visualize_method(self, mock_tight_layout, mock_subplots):
        """测试可视化方法"""
        # 模拟matplotlib对象
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # 设置颜色
        self.visualizer.model_colors = ['red', 'blue'] * 4

        # 执行可视化
        fig = self.visualizer._visualize()

        # 验证调用
        mock_subplots.assert_called_once()
        mock_tight_layout.assert_called_once()
        self.assertEqual(fig, mock_fig)

    @patch('msprobe.core.common.megatron_utils.Rectangle')
    def test_draw_single_block(self, mock_rectangle):
        """测试绘制单个模型块"""
        # 模拟矩形对象
        mock_rect = MagicMock()
        mock_rectangle.return_value = mock_rect

        # 模拟坐标轴
        mock_ax = MagicMock()

        # 设置颜色
        self.visualizer.model_colors = ['red', 'blue']

        # 测试数据
        test_cases = [
            (0, 0, 0, ["Embedding", 1, 2, 3]),  # 嵌入层
            (1, 2, 0, [7, 8, 9, 10]),             # 中间层
            (0, 3, 0, [19, 20, 21, "OutputLayer"])  # 输出层
        ]

        for dp_idx, tp_idx, vpp_stage, layers in test_cases:
            with self.subTest(dp_idx=dp_idx, tp_idx=tp_idx, vpp_stage=vpp_stage):
                self.visualizer._draw_single_block(
                    mock_ax, dp_idx, tp_idx, vpp_stage, layers)

                # 验证矩形创建
                mock_rectangle.assert_called()
                mock_ax.add_patch.assert_called_with(mock_rect)

                # 验证文本添加
                mock_ax.add_patch.assert_called()

                # 重置mock调用计数
                mock_rectangle.reset_mock()
                mock_ax.reset_mock()

    @patch('matplotlib.patches.Rectangle')
    def test_legend_creation(self, mock_rectangle):
        """测试图例创建"""
        # 模拟坐标轴
        mock_ax = MagicMock()

        # 设置颜色
        self.visualizer.model_colors = ['red', 'blue', 'green']
        self.visualizer.layout['num_model_copies'] = 3

        # 执行图例添加
        self.visualizer._add_legend(mock_ax)

        # 验证图例创建
        mock_ax.legend.assert_called_once()
        call_args = mock_ax.legend.call_args
        self.assertEqual(len(call_args[1]['handles']), 3)  # 3个图例元素

    @patch.object(ModelParallelismVisualizer, '_visualize')
    @patch('matplotlib.pyplot.close')
    def test_save_method(self, mock_close, mock_visualize):
        """测试保存方法"""
        # 模拟图表对象
        mock_fig = MagicMock()
        mock_visualize.return_value = mock_fig

        # 临时文件路径
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # 执行保存
            self.visualizer.save(tmp_path)

            # 验证调用
            mock_visualize.assert_called_once()
            mock_fig.savefig.assert_called_once_with(
                tmp_path, dpi=300, bbox_inches='tight'
            )
            mock_close.assert_called_once_with(mock_fig)

        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_edge_case_empty_models(self):
        """测试空模型配置的边缘情况"""
        empty_rank_models = {}
        empty_groups_dict = {"tp": [], "dp": [], "pp": []}

        # 应该能正常初始化
        visualizer = ModelParallelismVisualizer(
            self.args, empty_groups_dict, empty_rank_models
        )

        self.assertIsNotNone(visualizer)
        self.assertEqual(visualizer.layout['num_tp'], 0)

    def test_axes_setup(self):
        """测试坐标轴设置"""
        mock_ax = MagicMock()

        # 执行坐标轴设置
        self.visualizer._setup_axes(mock_ax)

        # 验证坐标轴标签设置
        mock_ax.set_xlabel.assert_called_with(
            'Virtual Pipeline Stage', fontsize=12)
        mock_ax.set_ylabel.assert_called_with('TP Group', fontsize=12)
        mock_ax.set_title.assert_called_once()
        mock_ax.set_xlim.assert_called_with(-0.5, 0.5)  # vpp_size=1
        mock_ax.set_ylim.assert_called_with(-0.5, 7.5)  # num_tp=8
        mock_ax.grid.assert_called_with(True, linestyle='--', alpha=0.3)

    def test_ticks_setup(self):
        """测试刻度设置"""
        mock_ax = MagicMock()

        # 执行刻度设置
        self.visualizer._setup_ticks(mock_ax)

        # 验证X轴刻度
        mock_ax.set_xticks.assert_called_with(range(0, 1))  # vpp_size=1
        mock_ax.set_xticklabels.assert_called_with(['Stage 0'])

        # 验证Y轴刻度
        mock_ax.set_yticks.assert_called_with(range(0, 8))  # num_tp=8
        # Y轴标签应该包含TP组信息
        yticklabels_call = mock_ax.set_yticklabels.call_args[0][0]
        self.assertEqual(len(yticklabels_call), 8)

    def test_virtual_pipeline_configuration(self):
        """测试虚拟流水线配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vpp_args = ParallelConfig(
                world_size=16,
                num_layers=96,
                tensor_parallel_size=4,
                pipeline_parallel_size=4,
                virtual_pipeline_parallel_size=4,
                output_path=temp_dir
            )

        vpp_groups_dict = {
            "tp": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            "dp": [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
            "pp": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        }

        vpp_rank_models = {
            i: [["Layer"] * 6 for _ in range(4)]  # 每个rank 4个虚拟阶段
            for i in range(16)
        }

        visualizer = ModelParallelismVisualizer(
            vpp_args, vpp_groups_dict, vpp_rank_models
        )

        self.assertEqual(visualizer.layout['vpp_size'], 4)
        self.assertEqual(visualizer.layout['num_tp'], 4)

    def test_single_device_configuration(self):
        """测试单设备配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            single_args = ParallelConfig(
                world_size=1,
                num_layers=12,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                output_path=temp_dir
            )

        single_groups_dict = {
            "tp": [[0]],
            "dp": [[0]],
            "pp": [[0]]
        }

        single_rank_models = {
            0: [["Embedding", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "OutputLayer"]]}

        visualizer = ModelParallelismVisualizer(
            single_args, single_groups_dict, single_rank_models
        )

        self.assertEqual(visualizer.layout['num_tp'], 1)
        self.assertEqual(visualizer.layout['num_model_copies'], 1)
        self.assertEqual(visualizer.layout['vpp_size'], 1)


class TestIntegration(unittest.TestCase):
    """测试集成功能"""

    @patch('msprobe.core.common.megatron_utils.ModelParallelismVisualizer')
    @patch('msprobe.core.common.megatron_utils.ModelLayerSimulator')
    @patch('msprobe.core.common.megatron_utils.RankGroupGenerator')
    def test_plot_model_parallelism(
        self,
        mock_generator,
        mock_simulator,
        mock_visualizer_class,
    ):
        mock_generator.return_value.generate_all_ranks.return_value = {
            "tp": [[0, 1]],
            "dp": [[0], [1]],
            "pp": [[0, 1]]
        }

        mock_simulator.return_value.generate_all_models.return_value = {
            0: [[1, 2, 3]],
            1: [[4, 5, 6]]
        }

        mock_visualizer_instance = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer_instance

        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=2,
                num_layers=6,
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                output_path=temp_dir
            )

        plot_model_parallelism(config)

        mock_generator.assert_called_once_with(1, 1, 1, 2, 1, 'tp-cp-ep-dp-pp')
        mock_visualizer_instance.save.assert_called_once_with(config.output_path)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_minimal_configuration(self):
        """测试最小配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=1,
                num_layers=1,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                output_path=temp_dir
            )

        self.assertEqual(config.data_parallel_size, 1)
        self.assertEqual(config.virtual_pipeline_parallel_size, 1)

    def test_large_scale_configuration(self):
        """测试大规模配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParallelConfig(
                world_size=1024,
                num_layers=256,  # 最大值
                tensor_parallel_size=16,
                pipeline_parallel_size=16,
                output_path=temp_dir
            )

        self.assertEqual(config.data_parallel_size, 4)  # 1024/(16 * 16)=4

    def test_invalid_configurations(self):
        """测试无效配置"""
        # 超过最大world_size
        with self.assertRaises(ValueError):
            ParallelConfig(
                world_size=1025,  # 超过1024
                num_layers=24,
                tensor_parallel_size=2,
                pipeline_parallel_size=2
            )

        # 超过最大层数
        with self.assertRaises(ValueError):
            ParallelConfig(
                world_size=8,
                num_layers=257,  # 超过256
                tensor_parallel_size=2,
                pipeline_parallel_size=2
            )

