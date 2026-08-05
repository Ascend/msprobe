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

# pylint: disable=duplicate-code  # 跨框架（mindspore/pytorch）实现相似是设计决定

from mindspore import mint, ops, _no_grad
from mindspore import Tensor
from mindspore import dtype as mstype

from msprobe.core.common.log import logger


@_no_grad()
def square_sum(x: Tensor):
    return (x * x).sum()


@_no_grad()
def get_min(x: Tensor):
    return mint.min(x)


@_no_grad()
def get_mean(x: Tensor):
    return mint.mean(x.astype(mstype.float32))


@_no_grad()
def get_norm(x: Tensor):
    norm_func = mint.norm if hasattr(mint, "norm") else ops.norm
    return norm_func(x.astype(mstype.float32))


@_no_grad()
def get_max(x: Tensor):
    return mint.max(x)


@_no_grad()
def get_zeros(x: Tensor, eps: float):
    if x.numel() == 0:
        return Tensor(float('nan'))
    return mint.sum(mint.abs(x) < eps) / x.numel()


@_no_grad()
def get_nans(t):
    return ops.isnan(t.astype(mstype.float32)).sum()


def get_shape(t):
    return t.shape


def get_dtype(t):
    return t.dtype


FUNC_MAP = {
    "min": get_min,
    "max": get_max,
    "mean": get_mean,
    "norm": get_norm,
    "nans": get_nans,
    "zeros": get_zeros,
    "shape": get_shape,
    "dtype": get_dtype,
}


def max_eigenvalue(input_tensor: Tensor, num_iterations=3):
    input_tensor = input_tensor.float()
    try:
        check_tensor_dim(input_tensor, 2)
    except (TypeError, ValueError) as e:
        logger.warning(f"calcute max eigenvalue failed, {e}")
        return Tensor(0)
    in_features = input_tensor.shape[1]
    u_tensor = ops.randn(in_features)
    u_norm = u_tensor.norm()
    if u_norm == 0:
        return Tensor(0)
    u_tensor /= u_tensor.norm()
    input_seq = ops.matmul(input_tensor.T, input_tensor)
    for _ in range(num_iterations):
        v_tensor = ops.matmul(input_seq, u_tensor)
        spectral_norm = ops.matmul(v_tensor.T, u_tensor)
        v_norm = v_tensor.norm()
        if v_norm > 0:
            u_tensor = v_tensor / v_norm
        else:
            spectral_norm = Tensor(0)
            break
    return spectral_norm.sqrt()


def check_tensor_dim(tensor, n):
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input must be a mindspore Tensor, but got {type(tensor)} instead.")
    if len(tensor.shape) < n:
        raise ValueError(
            f"tensor dim must be at least {n} dimensions.Got shape: {tuple(tensor.shape)} with {tensor.dim()} dims"
        )


def cal_entropy(qk_tensor: Tensor, mask=None):
    try:
        check_tensor_dim(qk_tensor, 2)
    except (TypeError, ValueError) as e:
        logger.warning(f"calculate entropy failed, {e}")
        return Tensor(0), Tensor(0)
    if mask is None:
        mask = ops.tril(ops.ones((qk_tensor.shape[1], qk_tensor.shape[1])))
    qk_tensor = qk_tensor - ops.amax(qk_tensor, axis=1, keepdims=True)
    qk_tensor = qk_tensor.masked_fill(mask == 0, float('-inf'))
    softmax_qkt = ops.softmax(qk_tensor.float(), axis=1)
    softmax_max = ops.mean(ops.amax(softmax_qkt, axis=1))
    entropy = ops.mean(-ops.nansum(softmax_qkt * ops.log(softmax_qkt), axis=1))
    return entropy, softmax_max


def cal_stable_rank(weight: Tensor):
    eig = max_eigenvalue(weight)
    if eig == Tensor(0):
        return Tensor(0), Tensor(0)
    f_norm = ops.norm(weight, ord='fro')
    return f_norm / eig, eig


@_no_grad()
def cal_router_weight_similarity(weight: Tensor):
    """Compute average cosine similarity between expert router weight columns.

    Args:
        weight: Router weight matrix. Expected shape (hidden_dim, num_experts) or
            (num_experts, hidden_dim). When the first dim is smaller than the second
            (i.e. num_experts < hidden_dim, the typical case for nn.Linear.weight),
            the input is transposed so each column represents one expert.

    Returns:
        Tensor: Scalar mean cosine similarity between expert weight columns.
            Returns 0 on invalid input or when there are fewer than 2 experts.
    """
    try:
        check_tensor_dim(weight, 2)
    except (TypeError, ValueError) as e:
        logger.warning(f"calculate router weight similarity failed, {e}")
        return Tensor(0.0, dtype=mstype.float32)
    if weight.dim() != 2:
        logger.warning(f"calculate router weight similarity failed, expected 2D tensor, got {weight.dim()}D.")
        return Tensor(0.0, dtype=mstype.float32)
    hidden_dim, num_experts = weight.shape
    # Hidden dim is usually larger than num_experts; transpose if needed so that
    # each column is one expert's weight vector.
    if hidden_dim < num_experts:
        weight = weight.T
        hidden_dim, num_experts = weight.shape
    if num_experts < 2:
        return Tensor(0.0, dtype=mstype.float32)
    weight = weight.astype(mstype.float32)
    col_norms = ops.sqrt(ops.sum(weight * weight, axis=0, keepdims=True) + 1e-12)
    normalized_w = weight / col_norms
    cos_sim = ops.matmul(normalized_w.T, normalized_w)
    # 取严格上三角（diagonal=1 排除对角线），sum 后除以元素数 = 平均余弦相似度
    upper_tri = ops.triu(cos_sim, diagonal=1)
    upper_count = num_experts * (num_experts - 1) // 2
    return ops.sum(upper_tri) / upper_count


@_no_grad()
def cal_per_token_expert_entropy(logits: Tensor):
    """Compute mean per-token entropy of the softmax distribution over experts.

    Args:
        logits: Router output logits of shape (..., num_experts). The last dim is
            treated as the expert dimension; all leading dims are flattened into
            the token dimension.

    Returns:
        Tensor: Scalar mean entropy across all tokens. Returns 0 on invalid input
            or when there are no tokens.
    """
    try:
        check_tensor_dim(logits, 1)
    except (TypeError, ValueError) as e:
        logger.warning(f"calculate per-token expert entropy failed, {e}")
        return Tensor(0.0, dtype=mstype.float32)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    num_experts = logits.shape[-1]
    # 用 tensor.reshape 方法而非 ops.reshape，规避 MindSpore + numpy 2.x 下 pyboost_reshape 的兼容问题
    logits = logits.astype(mstype.float32).reshape(-1, num_experts)
    n_token = logits.shape[0]
    if n_token == 0:
        return Tensor(0.0, dtype=mstype.float32)
    gates = ops.softmax(logits, axis=-1)
    # 用 nansum 处理被 -inf mask 的专家：0 * log(0) = NaN，nansum 视为 0；全 -inf 行经 softmax 为 NaN，nansum 同样视为 0
    entropy = -ops.nansum(gates * ops.log(gates), axis=-1)
    return ops.sum(entropy) / n_token


def cal_qkt(q_h: Tensor, k_h: Tensor, order="s,b,h,d"):
    # q_h shape is (s, b, h, d)
    try:
        check_tensor_dim(q_h, 4)
        check_tensor_dim(k_h, 4)
    except (TypeError, ValueError) as e:
        logger.warning(f"calculatee qkt failed, {e}")
        return Tensor(0)
    if order == "s,b,h,d":
        qkt = ops.matmul(q_h[:, 0, 0, :], k_h[:, 0, 0, :].t()) / q_h.shape[-1] ** 0.5
    elif order == "b,s,h,d":
        qkt = ops.matmul(q_h[0, :, 0, :], k_h[0, :, 0, :].t()) / q_h.shape[-1] ** 0.5
    else:
        logger.warning("Calculate qk tensor failed: Order unsupported.")
        qkt = Tensor(0)
    return qkt
