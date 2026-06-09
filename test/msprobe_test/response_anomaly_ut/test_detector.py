import numpy as np
import unittest
from msprobe.response_anomaly.detector import (
    ILLDetector,
    DetectionResult,
    RepetitionCounters,
    _resolve_model_name,
)


class DetectorForTest(ILLDetector):
    """用于测试 detector, 重写 __init__ 避免依赖配置文件"""

    def __init__(self):
        # 直接设置参数，不加载配置文件
        self.window_size = 128
        self.stride = 64
        self.topk = 20

        self.rare_explogp_sum_thresh = 0.4
        self.rare_cat_thresh = 2
        self.rare_top1_logp_thresh = -6

        self.garbled_top1_logp_thresh = -5
        self.garbled_window_ratio = 0.2
        self.garbled_window_thresh = 0

        self.repet_n = 3
        self.repet_distinct_n_thresh = 0.2
        self.repet_logp_thresh = -0.2
        self.w_std_threshold = 1e-12
        self.acf_threshold = 0.65
        self.acf_harmonic_threshold = 0.325
        self.acf_logp_thresh = -0.2
        self.acf_min_period = 3
        self.acf_max_period = 42
        self.linalg_logp_thresh = 0.9

        self.single_window_thresh = 14
        self.multi_window_thresh = 2

        self.mtype2vocab = {}
        self.tk2cat_path = None

        self._garbled_count = 0


class TestResolveModelName(unittest.TestCase):
    """测试 _resolve_model_name 函数"""

    def setUp(self):
        self.know_models = {"qwen-7b", "qwen-14b", "llama-2-7b", "baichuan2-7b"}

    def test_exact_match(self):
        """测试精准匹配"""
        model_config = "qwen7b"
        result = _resolve_model_name(model_config, self.know_models)
        self.assertEqual(result, "qwen-7b")

    def test_case_insensitive(self):
        """测试忽略大小写"""
        model_config = "QWEN-7B"
        result = _resolve_model_name(model_config, self.know_models)
        self.assertEqual(result, "qwen-7b")

    def test_with_separators(self):
        """测试忽略分隔符"""
        model_config = "qwen_7b"
        result = _resolve_model_name(model_config, self.know_models)
        self.assertEqual(result, "qwen-7b")

    def test_partial_match(self):
        """测试部分匹配（包含关系）"""
        model_config = "qwen-7b-chat"
        result = _resolve_model_name(model_config, self.know_models)
        self.assertEqual(result, "qwen-7b")

    def test_dict_config_match(self):
        """测试字典格式的 model_config"""
        model_config = {"model_name": "qwen-7b"}
        result = _resolve_model_name(model_config, self.know_models)
        self.assertEqual(result, "qwen-7b")

    def test_none_config(self):
        """测试 None 配置"""
        result = _resolve_model_name(None, self.know_models)
        self.assertIsNone(result)

    def test_none_match(self):
        """测试无匹配"""
        model_config = "unknown-model"
        result = _resolve_model_name(model_config, self.know_models)
        self.assertIsNone(result)


class TestHelperFunctions(unittest.TestCase):
    """测试辅助函数"""

    def test_get_ngrams(self):
        """测试 N-gram 生成"""
        mock = DetectorForTest()
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        ngrams = mock.get_ngrams(tokens)
        self.assertEqual(len(ngrams), 3)
        self.assertTrue(np.array_equal(ngrams[0], [1, 2, 3]))
        self.assertTrue(np.array_equal(ngrams[1], [2, 3, 4]))
        self.assertTrue(np.array_equal(ngrams[2], [3, 4, 5]))

    def test_get_ngrams_short(self):
        """测试短序列的 N-gram"""
        mock = DetectorForTest()
        tokens = np.array([1, 2], dtype=np.int64)
        ngrams = mock.get_ngrams(tokens)
        self.assertEqual(len(ngrams), 0)

    def test_get_distinct_n(self):
        """测试 distinct N-gram 计算"""
        mock = DetectorForTest()

        # 完全重复序列 distinct_n = 1/3 (3个 n-gram 完全相同)
        tokens = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        distinct = mock.get_distinct_n(tokens)
        self.assertAlmostEqual(distinct, 0.333, places=2)

        # 完全不同的序列 distinct_n = 3/3 (3个 n-gram 完全不相同)
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        distinct = mock.get_distinct_n(tokens)
        self.assertEqual(distinct, 1.0)


class TestDetectionResult(unittest.TestCase):
    """测试检测结果数据结构"""

    def test_default_result(self):
        """测试默认检测结果"""
        result = DetectionResult()
        self.assertFalse(result.is_ill)
        self.assertEqual(result.ill_type, 0)

    def test_ill_result(self):
        """测试异常检测结果"""
        result = DetectionResult(is_ill=True, ill_type=1)
        self.assertTrue(result.is_ill)
        self.assertEqual(result.ill_type, 1)


class TestRepetitionCounters(unittest.TestCase):
    """测试重复检测计数器"""

    def test_default_counters(self):
        """测试默认计数器"""
        counters = RepetitionCounters()
        self.assertEqual(counters.both, 0)
        self.assertEqual(counters.acf_only, 0)
        self.assertEqual(counters.traj_only, 0)


class TestSlidingWindow(unittest.TestCase):
    """测试滑窗生成器"""

    def setUp(self):
        """每个测试前创建 DetectorForTest"""
        self.mock = DetectorForTest()
        self.mock.window_size = 4
        self.mock.stride = 2

    def test_sliding_window_full(self):
        """测试完整滑窗"""
        seq = [1, 2, 3, 4, 5, 6]
        windows = list(self.mock.sliding_window(seq))
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], (0, [1, 2, 3, 4]))
        self.assertEqual(windows[1], (2, [3, 4, 5, 6]))
        self.assertEqual(windows[2], (4, [5, 6]))

    def test_sliding_window_short(self):
        """测试短序列滑窗"""
        seq = [1, 2]
        windows = list(self.mock.sliding_window(seq))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], (0, [1, 2]))


class TestACFDetector(unittest.TestCase):
    """测试 ACF 检测器"""

    def setUp(self):
        self.mock = DetectorForTest()

    def test_acf_constant_signal(self):
        """测试常量信号的 ACF"""
        logprobs = np.array([0.5] * 128)
        # 常量信号的 std 为0， 应该返回 False
        result = self.mock._acf_detector(logprobs)
        self.assertFalse(result)

    def test_acf_random_signal(self):
        np.random.seed(42)
        logprobs = np.random.randn(128) * 0.1 + 0.5
        result = self.mock._acf_detector(logprobs)
        # 随机信号不应该被检测为重复
        self.assertFalse(result)

    def test_acf_periodic_signal(self):
        """测试周期性信号的 ACF"""
        t = np.arange(128)
        singal = 0.5 * np.sin(2 * np.pi * t / 10)
        noise = np.random.normal(0, 0.005, 128)
        logprobs = -0.05 + 0.05 * singal + noise
        result = self.mock._acf_detector(logprobs)
        self.assertTrue(result)


class TestTrajectoryDetector(unittest.TestCase):
    """测试轨迹检测器"""

    def setUp(self):
        self.mock = DetectorForTest()
        self.mock.repet_distinct_n_thresh = 0.2
        self.mock.repet_logp_thresh = -0.2

    def test_repetitive_sequence(self):
        """测试重复序列"""
        logprobs = np.array([-0.1] * 128)
        tokens = np.array([1] * 128, dtype=np.int64)
        result = self.mock._trajectory_detector(logprobs, tokens)
        self.assertTrue(result)

    def test_diverse_sequence(self):
        """测试多样化序列"""
        logprobs = np.array([-0.1] * 128)
        tokens = np.arange(128, dtype=np.int64)
        result = self.mock._trajectory_detector(logprobs, tokens)  # 完全不同的序列
        self.assertFalse(result)


class TestGarbledDetection(unittest.TestCase):
    """测试乱码检测"""

    def setUp(self):
        self.mock = DetectorForTest()
        self.mock.topk = 20
        self.mock.rare_explogp_sum_thresh = 0.4
        self.mock.garbled_top1_logp_thresh = -5
        self.mock.garbled_window_ratio = 0.6

        self.tk2cat = {str(i): i % 5 for i in range(1000, 2000)}
        self.vocab_size = 2000

    def test_normal_logprobs_notk2cat(self):
        """测试正常概率分布,无词表情况"""
        window_topk_logprobs = [
            {1000 + j: -0.1 for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        result = self.mock._detect_garbled(
            window_topk_logprobs, window_logprobs, None, 0
        )
        self.assertFalse(result)

    def test_normal_logprobs_tk2cat(self):
        """测试正常概率分布,有词表情况"""
        window_topk_logprobs = [
            {1000 + j: -0.1 for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        result = self.mock._detect_garbled(
            window_topk_logprobs, window_logprobs, self.tk2cat, self.vocab_size
        )
        self.assertFalse(result)

    def test_non_tk2cat_detect(self):
        """测试乱码检测,无词表情况"""
        window_topk_logprobs = [
            {1000 + j: -10.0 for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        result = self.mock._detect_garbled(
            window_topk_logprobs, window_logprobs, None, 0
        )
        self.assertTrue(result)

    def test_tk2cat_detect(self):
        """测试乱码检测,有词表情况"""
        window_topk_logprobs = [
            {1000 + j: -10.0 for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        result = self.mock._detect_garbled(
            window_topk_logprobs, window_logprobs, self.tk2cat, self.vocab_size
        )
        self.assertTrue(result)


class TestRareCharacterDetection(unittest.TestCase):
    """检测生僻字检测"""

    def setUp(self):
        self.mock = DetectorForTest()
        self.mock.topk = 20
        self.mock.rare_explogp_sum_thresh = 0.4
        self.mock.rare_cat_thresh = 2
        self.mock.rare_top1_logp_thresh = -6
        # 模拟 token2category 映射
        self.tk2cat = {str(i): i % 5 for i in range(1000, 2000)}
        self.vocab_size = 2000

    def test_normal_characters_tk2cat(self):
        """测试正常字符,有词表情况"""
        # 正常概率分布
        window_topk_logprobs = [
            {1000 + j: -0.1 * j for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        rare_flag, _ = self.mock._detect_rare_character(
            window_topk_logprobs, window_logprobs, self.tk2cat, self.vocab_size
        )
        self.assertFalse(rare_flag)

    def test_normal_characters_notk2cat(self):
        """测试正常字符,无词表情况"""
        # 正常概率分布
        window_topk_logprobs = [
            {1000 + j: -0.1 * j for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        rare_flag, _ = self.mock._detect_rare_character(
            window_topk_logprobs, window_logprobs, None, 0
        )
        self.assertFalse(rare_flag)

    def test_rare_characters_tk2cat(self):
        """测试生僻字,有词表情况"""
        window_topk_logprobs = [
            {1000 + j: -5.0 for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        rare_flag, _ = self.mock._detect_rare_character(
            window_topk_logprobs, window_logprobs, self.tk2cat, self.vocab_size
        )
        self.assertTrue(rare_flag)

    def test_rare_characters_notk2cat(self):
        """测试生僻字,无词表情况"""
        window_topk_logprobs = [
            {1000 + j: -8.0 for j in range(self.mock.topk)} for _ in range(128)
        ]
        window_logprobs = np.array(
            [list(item.values()) for item in window_topk_logprobs]
        )
        rare_flag, _ = self.mock._detect_rare_character(
            window_topk_logprobs, window_logprobs, None, 0
        )
        self.assertTrue(rare_flag)


class TestDetectorIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """创建 DetectorForTest 用于集成测试"""
        self.mock = DetectorForTest()

    def test_empty_input(self):
        """测试空输入"""
        result = self.mock.detector([], [])
        self.assertFalse(result.is_ill)

    def test_short_sequence(self):
        """测试短序列"""
        topk_logprobs = [{1000: -0.1, 1001: -0.2} for _ in range(20)]
        tokens = [1000] * 20

        # 短序列 正常 logprob 不应触发异常，返回正常
        result = self.mock.detector(topk_logprobs, tokens, None)
        self.assertFalse(result.is_ill)

    def test_normal_sequence(self):
        """测试正常序列"""
        np.random.seed(42)
        seq_len = 200
        topk_logprobs = [
            {1000 + j: -0.1 * j + np.random.randn() * 0.01 for j in range(20)}
            for _ in range(seq_len)
        ]
        tokens = list(range(1000, 1000 + seq_len))
        result = self.mock.detector(topk_logprobs, tokens, None)
        # 正常序列不应该被检测为异常
        self.assertFalse(result.is_ill)

    def test_repetitive_sequence(self):
        """测试重复序列"""
        seq_len = 1000
        topk_logprobs = [{1000: -0.01, 1001: -0.1} for _ in range(seq_len)]
        tokens = [1000] * seq_len
        result = self.mock.detector(topk_logprobs, tokens, None)

        self.assertTrue(result.is_ill)
        self.assertEqual(result.ill_type, 3)


class TestBatchProcessing(unittest.TestCase):
    """测试批量处理"""

    def setUp(self):
        self.mock = DetectorForTest()

    def test_batch_normal(self):
        """测试批量正常序列"""
        seq_len = 100
        topk_logprobs = [
            [{1000 + j: -0.1 * j for j in range(20)} for _ in range(seq_len)]
            for _ in range(5)
        ]
        tokens = [list(range(1000, 1000 + seq_len)) for _ in range(5)]
        result = self.mock.run(topk_logprobs, tokens, None)
        self.assertEqual(len(result), 5)

        for res in result:
            self.assertFalse(res[0])

    def test_batch_mixed(self):
        """测试批量混合序列"""
        seq_len = 200
        # 3 个正常 + 2 个重复
        topk_logprobs = []
        tokens = []
        for i in range(5):
            if i < 3:
                topk_logprobs.append(
                    [{1000 + j: -0.1 * j for j in range(20)} for _ in range(seq_len)]
                )
                tokens.append(list(range(1000, 1000 + seq_len)))
            else:
                topk_logprobs.append(
                    [{1000: -0.01, 1001: -0.1} for _ in range(seq_len)]
                )
                tokens.append([1000] * seq_len)

        result = self.mock.run(topk_logprobs, tokens, None)
        self.assertEqual(len(result), 5)

        # 前 3 个应该为正常，后 2 个可能是异常(取决于阈值)
        self.assertFalse(result[0][0])
        self.assertFalse(result[1][0])
        self.assertFalse(result[2][0])
