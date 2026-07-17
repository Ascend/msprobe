import unittest
from unittest.mock import patch

from mindspore import mint, ops
from mindspore import Tensor
from mindspore import dtype as mstype

from msprobe.mindspore.monitor.features import max_eigenvalue, cal_entropy, cal_qkt, cal_stable_rank


class TestMathFunctions(unittest.TestCase):
    @staticmethod
    def _dominant_randn(in_features):
        vector = [0.0] * int(in_features)
        vector[0] = 1.0
        return Tensor(vector, dtype=mstype.float32)

    def test_max_eigenvalue(self):
        """测试最大特征值计算"""
        # 创建已知特征值的矩阵
        A = ops.diag(Tensor([3.0, 2.0, 1.0]))

        # 测试不同迭代次数
        with patch("msprobe.mindspore.monitor.features.ops.randn", side_effect=self._dominant_randn):
            eigval = max_eigenvalue(A, num_iterations=5)
            self.assertAlmostEqual(eigval.item(), 3.0, delta=0.1)

        # 测试全零矩阵
            zero_matrix = ops.zeros((3, 3))
            eigval = max_eigenvalue(zero_matrix)
            self.assertAlmostEqual(eigval.item(), 0.0)

    def test_cal_entropy(self):
        """测试注意力熵计算"""
        # 创建简单的注意力分数
        qk = Tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

        # 无mask
        entropy, softmax_max = cal_entropy(qk)
        self.assertAlmostEqual(entropy, 0.4715, delta=0.1)
        self.assertAlmostEqual(softmax_max, 0.7988, delta=0.1)

        # 带mask 和默认生成相同
        mask = Tensor([[1, 0, 0],
                             [1, 1, 0],
                             [1, 1, 1]], dtype=mstype.float32)
        entropy, softmax_max = cal_entropy(qk, mask)
        self.assertAlmostEqual(entropy, 0.4715, delta=0.1)
        self.assertAlmostEqual(softmax_max, 0.7988, delta=0.1)

    @patch("msprobe.mindspore.monitor.features.logger")
    def test_cal_qkt(self, mock_logger):
        """测试QK^T计算"""
        # 测试s,b,h,d顺序
        q = ops.randn((10, 2, 4, 8))  # [s, b, h, d]
        k = ops.randn((10, 2, 4, 8))  # [s, b, h, d]
        q_batch = ops.randn((2, 10, 4, 8))  # [b, s, h, d]
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
        with patch("msprobe.mindspore.monitor.features.ops.randn", side_effect=self._dominant_randn):
            A = ops.diag(Tensor([3.0, 2.0, 1.0]))
            sr, eig = cal_stable_rank(A)

        # 验证Frobenius范数
            fro_norm = ops.norm(A, ord='fro')
            self.assertAlmostEqual(sr, fro_norm / 3.0, delta=.5)  # 最大特征值为3

        # 测试正交矩阵
            ortho = ops.eye(5)
            sr, eig = cal_stable_rank(ortho)
            self.assertAlmostEqual(sr, Tensor(2.23/1), delta=.5)  # F范数应为2.23
            self.assertAlmostEqual(eig, Tensor(1.0), delta=.1)  # 特征值应为1


if __name__ == '__main__':
    unittest.main()
