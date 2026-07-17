import unittest
from unittest.mock import patch

import torch
from msprobe.pytorch.monitor.features import square_sum, get_min, get_mean, get_norm, get_max, get_zeros, \
    get_sign_matches, eff_rank, mNTK, power_iteration, lambda_max_subsample, cal_histc, get_nans, check_tensor_dim
from msprobe.pytorch.monitor.features import max_eigenvalue, cal_entropy, cal_qkt, cal_stable_rank


class TestMathFunctions(unittest.TestCase):
    @staticmethod
    def _dominant_randn(in_features):
        vector = torch.zeros(in_features)
        vector[0] = 1.0
        return vector

    def test_square_sum(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = square_sum(tensor)
        self.assertEqual(result, 14.0)

    def test_get_min(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_min(tensor)
        self.assertEqual(result, 1.0)

    def test_get_mean(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_mean(tensor)
        self.assertAlmostEqual(result, 2.0, places=1)

    def test_get_norm(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_norm(tensor)
        self.assertTrue(torch.allclose(result, torch.tensor(3.7417, dtype=torch.float64), atol=1e-4))

    def test_get_max(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_max(tensor)
        self.assertEqual(result, 3.0)

    def test_get_zeros(self):
        tensor = torch.tensor([1e-10, 2e-10, 3e-10])
        result = get_zeros(tensor, eps=1e-10)
        res = torch.allclose(result, torch.tensor(0.), atol=1e-1)
        self.assertTrue(res)

    def test_get_sign_matches(self):
        tensor_x = torch.tensor([1.0, -1.0, 1.0])
        tensor_y = torch.tensor([1.0, 1.0, -1.0])
        result = get_sign_matches(tensor_x, tensor_y)
        res = torch.allclose(result, torch.tensor(0.3333), atol=1e-4)
        self.assertTrue(res)

    def test_eff_rank(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
        result = eff_rank(tensor)
        res = torch.allclose(result, torch.tensor(2), atol=1e-1)
        self.assertTrue(res)

    def test_mNTK(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super(MockModule, self).__init__()

            def forward(self, x):
                return x + 1

        module = MockModule()
        tensor = torch.tensor([1.0])
        result = mNTK(module, tensor)
        res = torch.allclose(result, torch.tensor([[1.]]), atol=1e-1)
        self.assertTrue(res)

    def test_power_iteration(self):
        a = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        eigval = power_iteration(a, num_iterations=10)
        self.assertAlmostEqual(eigval.item(), 2.0, places=2)

    def test_lambda_max_subsample(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super(MockModule, self).__init__()

            def forward(self, x):
                return x

        module = MockModule()
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lambda_max_subsample(module, tensor)
        res = torch.allclose(result, torch.tensor(1.0), atol=1e-1)
        self.assertTrue(res)

    def test_cal_histc(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cal_histc(tensor, bins_total=3, min_val=1.0, max_val=5.0)
        self.assertEqual(result.size(), (3,))

    def test_get_nans(self):
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        result = get_nans(tensor)
        self.assertEqual(result, 1)

    def test_check_tensor_dim(self):
        check_tensor_dim(torch.randn(2, 3, 4), 3)
        # TypeError
        with self.assertRaises(TypeError):
            check_tensor_dim(123, 2)
        # ValueError
        with self.assertRaises(ValueError):
            check_tensor_dim(torch.randn(2, 3), 3)

    def test_max_eigenvalue(self):
        """测试最大特征值计算"""
        # 创建已知特征值的矩阵
        A = torch.diag(torch.tensor([3.0, 2.0, 1.0]))

        # 测试不同迭代次数
        with patch("msprobe.pytorch.monitor.features.torch.randn", side_effect=self._dominant_randn):
            eigval = max_eigenvalue(A, num_iterations=5)
            self.assertAlmostEqual(eigval.item(), 3.0, delta=0.1)

        # 测试全零矩阵
            zero_matrix = torch.zeros(3, 3)
            eigval = max_eigenvalue(zero_matrix)
            self.assertAlmostEqual(eigval.item(), 0.0)

    def test_cal_entropy(self):
        """测试注意力熵计算"""
        # 创建简单的注意力分数
        qk = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        # 无mask
        entropy, softmax_max = cal_entropy(qk)
        self.assertAlmostEqual(entropy, 0.4715, delta=0.1)
        self.assertAlmostEqual(softmax_max, 0.7988, delta=0.1)

        # 带mask 和默认生成相同
        mask = torch.tensor([[1, 0, 0],
                             [1, 1, 0],
                             [1, 1, 1]], dtype=torch.float)
        entropy, softmax_max = cal_entropy(qk, mask)
        self.assertAlmostEqual(entropy, 0.4715, delta=0.1)
        self.assertAlmostEqual(softmax_max, 0.7988, delta=0.1)

    @patch("msprobe.pytorch.monitor.features.logger")
    def test_cal_qkt(self, mock_logger):
        """测试QK^T计算"""
        # 测试s,b,h,d顺序
        q = torch.randn(10, 2, 4, 8)  # [s, b, h, d]
        k = torch.randn(10, 2, 4, 8)  # [s, b, h, d]
        q_batch = torch.randn(2, 10, 4, 8)  # [b, s, h, d]
        qkt = cal_qkt(q, k, order="s,b,h,d")
        self.assertEqual(qkt.shape, (10, 10))  # [s, s]

        # 测试b,s,h,d顺序
        qkt = cal_qkt(q_batch, q_batch, order="b,s,h,d")
        self.assertEqual(qkt.shape, (10, 10))  # [s, s]

        # 测试无效顺序
        cal_qkt(q, k, order="invalid_order")
        mock_logger.warning.assert_called_with(
            "Calculate qk tensor failed: Order unsupported.")

    def test_cal_stable_rank(self):
        """测试谱半径计算"""
        # 创建已知谱半径的矩阵
        with patch("msprobe.pytorch.monitor.features.torch.randn", side_effect=self._dominant_randn):
            A = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
            sr, eig = cal_stable_rank(A)

        # 验证Frobenius范数
            fro_norm = torch.norm(A, p='fro')
            self.assertAlmostEqual(sr, fro_norm / 3.0, delta=.5)  # 最大特征值为3

        # 测试正交矩阵
            ortho = torch.eye(5)
            sr, eig = cal_stable_rank(ortho)
            self.assertAlmostEqual(sr, torch.tensor(2.23/1), delta=.5)  # F范数应为2.23
            self.assertAlmostEqual(eig, torch.tensor(1.0), delta=.1)  # 特征值应为1


if __name__ == '__main__':
    unittest.main()
