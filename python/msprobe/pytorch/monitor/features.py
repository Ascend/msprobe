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

import torch
from torch.autograd.functional import jacobian
from msprobe.pytorch.common.log import logger


@torch.no_grad()
def square_sum(x: torch.tensor):
    return (x * x).sum()


@torch.no_grad()
def get_min(x: torch.tensor):
    return torch.min(x)


@torch.no_grad()
def get_mean(x: torch.tensor):
    return torch.mean(x.to(torch.float64))


@torch.no_grad()
def get_norm(x: torch.tensor):
    return torch.norm(x.to(torch.float64), p=2)


@torch.no_grad()
def get_max(x: torch.tensor):
    return torch.max(x)


@torch.no_grad()
def get_zeros(x: torch.tensor, eps: float):
    if x.numel() == 0:
        return torch.tensor(float('nan'))
    return torch.sum(torch.abs(x) < eps) / x.numel()


@torch.no_grad()
def get_sign_matches(x: torch.tensor, y: torch.tensor):
    if y.numel() == 0:
        return torch.tensor(1.0)
    xs = x.sign()
    ys = y.sign()

    try:
        same_direction_ratio = ((xs * ys).sum() / ys.numel() + 1) / 2
    except RuntimeError as e:
        logger.info(f"RuntimeError: {e}")
        same_direction_ratio = torch.tensor(0.0)
    return same_direction_ratio


@torch.no_grad()
def eff_rank(param: torch.tensor, threshold=1e-10):
    U, S, Vh = torch.linalg.svd(param.float())
    rank = torch.sum(S > threshold)
    return rank


# modular neural tangent kernel
@torch.no_grad()
def mNTK(module: torch.nn.Module, x: torch.tensor):
    J_theta_l = jacobian(module, x)
    mntk = torch.matmul(J_theta_l, J_theta_l.t())
    return mntk


@torch.no_grad()
def power_iteration(a, num_iterations):
    b = torch.randn(a.size(1), 1)
    for _ in range(num_iterations):
        b = torch.matmul(a, b)
        b_norm = torch.norm(b)
        b = b / b_norm if b_norm != 0 else 0
    eigval = torch.matmul(torch.matmul(b.t(), a), b)
    return eigval


@torch.no_grad()
def lambda_max_subsample(module: torch.nn.Module, x: torch.tensor, num_iterations=100, subsample_size=None):
    mntk = mNTK(module, x)
    if subsample_size is None:
        subsample_size = min(mntk.size(0), mntk.size(1))
    idx = torch.randperm(mntk.size(0))[:subsample_size]
    subsampled = mntk[idx, :]
    subsampled = subsampled[:, idx]
    eigval = power_iteration(subsampled, num_iterations)
    return eigval


@torch.no_grad()
def cal_histc(tensor_cal, bins_total, min_val, max_val):
    return torch.histc(tensor_cal, bins=bins_total, min=min_val, max=max_val)


@torch.no_grad()
def get_nans(t):
    return torch.isnan(t).sum()


def check_tensor_dim(tensor, n):
    """检查张量维度是否大于n"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Input must be a PyTorch tensor. Got {type(tensor)} instead. Consider using torch.tensor() for conversion."
        )

    if tensor.dim() < n:
        raise ValueError(
            f"Tensor must have at least {n} dimensions. Got shape: {tuple(tensor.shape)} with {tensor.dim()} dims."
        )


@torch.no_grad()
def max_eigenvalue(input_tensor: torch.Tensor, num_iterations=3):
    input_tensor = input_tensor.float()
    try:
        check_tensor_dim(input_tensor, 2)
    except (TypeError, ValueError) as e:
        logger.warning(f"Calculate max eigenvalue failed: {e}")
        return torch.tensor(0)
    in_features = input_tensor.shape[1]
    u_tensor = torch.randn(in_features).to(input_tensor.device)
    u_norm = u_tensor.norm()
    if u_norm.item() == 0:
        return torch.tensor(0)
    u_tensor = u_tensor / u_tensor.norm()
    input_seq = torch.matmul(input_tensor.T, input_tensor)
    for _ in range(num_iterations):
        v_tensor = torch.matmul(input_seq, u_tensor)
        spectral_norm = torch.matmul(v_tensor.T, u_tensor)
        v_norm = v_tensor.norm()
        if v_norm > 0:
            u_tensor = v_tensor / v_norm
        else:
            spectral_norm = torch.tensor(0)
            break
    return spectral_norm.sqrt()


@torch.no_grad()
def cal_entropy(qk_tensor, mask=None):
    try:
        check_tensor_dim(qk_tensor, 2)
    except (TypeError, ValueError) as e:
        logger.warning(f"Calculate max eigenvalue failed: {e}")
        return torch.tensor(0), torch.tensor(0)
    if mask is None:
        mask = torch.tril(torch.ones(qk_tensor.shape[1], qk_tensor.shape[1])).to(qk_tensor.device)
    qk_tensor = qk_tensor - torch.amax(qk_tensor, dim=1, keepdim=True)
    qk_tensor = qk_tensor.masked_fill(mask == 0, float('-inf'))
    softmax_qkt = torch.nn.functional.softmax(qk_tensor.float(), dim=1)
    # softmax取QK矩阵最大值
    softmax_max = torch.mean(torch.amax(softmax_qkt, dim=1))
    # 只对 p > 0 的位置算 log，p=0 时贡献为 0, nansum 可能在部分设备不支持，用小算子替换
    log_p = torch.where(softmax_qkt > 0, torch.log(softmax_qkt), torch.tensor(0.0, device=softmax_qkt.device))
    entropy = torch.mean(-(softmax_qkt * log_p).sum(dim=1))
    return entropy, softmax_max


@torch.no_grad()
def cal_qkt(q_h, k_h, order="s,b,h,d"):
    # q_h shape is [s, b, h, d]
    try:
        check_tensor_dim(q_h, 4)
        check_tensor_dim(k_h, 4)
    except (TypeError, ValueError) as e:
        logger.warning(f"Calculate qk tensor failed: {e}")
        return torch.tensor(0)

    if order == "s,b,h,d":
        qkt = torch.matmul(q_h[:, 0, 0, :], k_h[:, 0, 0, :].t()) / q_h.shape[-1] ** 0.5
    elif order == "b,s,h,d":
        qkt = torch.matmul(q_h[0, :, 0, :], k_h[0, :, 0, :].t()) / q_h.shape[-1] ** 0.5
    else:
        logger.warning("Calculate qk tensor failed: Order unsupported.")
        qkt = torch.tensor(0)
    return qkt


@torch.no_grad()
def cal_stable_rank(weight: torch.Tensor):
    eig = max_eigenvalue(weight)
    if eig == torch.tensor(0):
        return torch.tensor(0), torch.tensor(0)
    f_norm = torch.norm(weight, p="fro")
    return f_norm / eig, eig
