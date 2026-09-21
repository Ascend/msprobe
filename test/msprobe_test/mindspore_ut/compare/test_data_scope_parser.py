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
from msprobe.core.compare.layer_mapping.data_scope_parser import (
    find_regard_scope,
    find_stack_func_list,
    get_dump_data_items,
    get_stack_in_lines,
)
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml


class TestModifyMapping(unittest.TestCase):

    def setUp(self):
        # 这里可以设置常量和初始化
        self.lines = [
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
            in _run_construct, \n output = self._run_forward_hook(inputs, output)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
            in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
            in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
            in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/utils.py, line 38, \
            in attn_mask_add, \n attention_scores = ops.add(",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/scale_mask_softmax.py, line 65, \
            in construct, \n masked_input = self.mask_func(x, mask) if mask is not None else x",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
            in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
            in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
            in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
            in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
            in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
            in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
            in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
            in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
            in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
            in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
        ]

        self.frame_func_lines = [
            "File /home/user/envs/python3.9/site-packages/mindspore/common/tensor.py, line 2465, \
            in copy, x = x / 1.0",
            "File /home/user/envs/python3.9/site-packages/mindformers/tensor_parallel/layers.py, line 1147, \
            in construct, \n masked_input = input_.copy() = self.vocab_start_index",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2455, \
            in _backward_hook_construct, \n outputs = self.construct(outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 494, \
            in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 733, \
            in __call__, \n return self._run_construct(*args, *kwargs)"
        ]

        self.simplified_lines = [
            "File /home/user/envs/python3.9/site-packages/mindspore/common/tensor.py, line 2465, \
            in copy, \n x = x / 1.0",
            "File /home/user/envs/python3.9/site-packages/mindformers/tensor_parallel/layers.py, line 1147, \
            in construct, \n masked_input = input_.copy() = self.vocab_start_index"
        ]

        self.pt_construct = {
            "Functional.max_pool2d.0.forward": "Module.pool1.MaxPool2d.forward.0",
            "Functional.conv2d.1.forward": "Module.conv2.Conv2d.forward.0",
            "Functional.linear.5.backward": "Module.fc3.Linear.backward.1",
            "Module.conv1.Conv2d.backward.1": None,
            "Functional.conv2d.2.forward": None
        }

        self.ms_construct = {
            "Functional.add.0.forward": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0",
            "Tensor.reshape.2.forward": "Cell.transformer_layers.0.attention.ParallelAttention.forward.0",
            "Functional.add.4.forward": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0"
        }

        self.ms_dump = {
            "data": {"Functional.add.0.forward": "",
                     "Functional.add.4.forward": "",
                     "Tensor.reshape.2.forward": ""
                     }
        }
        self.pt_dump = {
            "data": {
                "Functional.max_pool2d.0.forward": "",
                "Functional.conv2d.1.forward": "",
                "Functional.linear.5.backward": "",
                "Functional.conv2d.2.forward": "",
                "Module.conv1.Conv2d.backward.1": ""
            }
        }

        self.ms_stack = {
            "Functional.add.0.forward": [
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
                in _run_construct, \n output = self._run_forward_hook(inputs, output)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
                in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
                in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/utils.py, line 38, \
                in attn_mask_add, \n attention_scores = ops.add(",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/scale_mask_softmax.py, line 65, \
                in construct, \n masked_input = self.mask_func(x, mask) if mask is not None else x",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
            ],
            "Tensor.reshape.2.forward": [
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
                in _run_construct, \n output = self._run_forward_hook(inputs, output)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
                in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
                in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 367, \
                in construct, \n query = query.reshape(bs, seq_len, -1, self.head_dim).transpose((0, 2, 1, 3))",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
            ],
            "Functional.add.4.forward": [
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
                in _run_construct, \n output = self._run_forward_hook(inputs, output)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
                in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
                in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/utils.py, line 38, \
                in attn_mask_add, \n attention_scores = ops.add(",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/scale_mask_softmax.py, line 65, \
                in construct, \n masked_input = self.mask_func(x, mask) if mask is not None else x",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
            ]
        }

        self.pt_stack = {
            "Functional.max_pool2d.0.forward": [
                "File /home/user/envs/lib/python3.9/site-packages/msprobe/pytorch/hook_module/wrap_Functional.py, line 97, \
                in functional_op_template, \n return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/pooling.py, line 166, \
                in forward, \n return F.max_pool2d(input, self.kernel_size, self.stride,",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1568, \
                in _call_impl, \n result = foward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 28, in forward, \n x = self.pool1(x)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1527, \
                in _call_impl, \n return forward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 47, in train_one_epoch, \n output = model(image)",
                "File lenet.py, line 83, in TestLeNetE2E, \n train_one_epoch(step, steps, train_loader, \
                model, optimizer, criterion)",
                "File lenet.py, line 88, in <module>, \n TestLeNetE2E()"
            ],
            "Functional.conv2d.1.forward": [
                "File /home/user/envs/lib/python3.9/site-packages/msprobe/pytorch/hook_module/wrap_Functional.py, line 97, \
                in functional_op_template, \n return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/conv.py, line 456, \
                in _conv_forward, \n return F.conv2d(input, weight, bias, self.stride,",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/conv.py, line 460, \
                in forward, \n return self._conv_forward(input, self.weight, self.bias)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1568, \
                in _call_impl, \n result = foward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 29, in forward, \n x = F.relu(self.conv2(x))",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1527, \
                in _call_impl, \n return forward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 47, in train_one_epoch, \n output = model(image)",
                "File lenet.py, line 83, in TestLeNetE2E, \n train_one_epoch(step, steps, train_loader, \
                model, optimizer, criterion)",
                "File lenet.py, line 88, in <module>, \n TestLeNetE2E()"
            ]
        }

        self.pt_dump_source = {
        "task": "statistics",
        "level": "mix",
        "dump_data_dir": None,
        "data": {
            "Tensor.__add__.0.forward": {
                "input_args": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.int64",
                        "shape": [],
                        "Max": 423,
                        "Min": 423,
                        "Mean": 423,
                        "Norm": 423,
                        "requires_grad": False
                    },
                    {
                        "type": "int",
                        "value": 1
                    }
                ],
                "input_kwargs": {},
                "output": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.int64",
                        "shape": [],
                        "Max": 424,
                        "Min": 424,
                        "Mean": 424,
                        "Norm": 424,
                        "requires_grad": False
                    }
                ]
            },
            "Tensor.__add__.1.forward": {
                "input_args": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.int64",
                        "shape": [],
                        "Max": 423,
                        "Min": 423,
                        "Mean": 423,
                        "Norm": 423,
                        "requires_grad": False
                    },
                    {
                        "type": "int",
                        "value": 1
                    }
                ],
                "input_kwargs": {},
                "output": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.int64",
                        "shape": [],
                        "Max": 424,
                        "Min": 424,
                        "Mean": 424,
                        "Norm": 424,
                        "requires_grad": False
                    }
                ]
            },
                }
        }

        self.ms_dump_source = {
            "task": "statistics",
            "level": "mix",
            "dump_data_dir": None,
            "data": {
                # layer type
                "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0": {
                    "input_args": [],
                    "input_kwargs": {},
                    "output": [
                        {
                            "type": "mindspore.Tensor",
                            "dtype": "BFloat16",
                            "shape": [
                                1024,
                                1,
                                6144
                            ],
                            "Max": 0.421875,
                            "Min": -0.443359375,
                            "Mean": -0.0002346038818359375,
                            "Norm": 50.75
                        },
                        None
                    ]
                },
                # Mint Type
                "Mint.cos.0.forward": {
                    "input_args": [
                        {
                            "type": "mindspore.Tensor",
                            "dtype": "Float32",
                            "shape": [
                                4096,
                                1,
                                1,
                                128
                            ],
                            "Max": 4095.0,
                            "Min": 0.0,
                            "Mean": 238.66024780273438,
                            "Norm": 427910.46875
                        }
                    ],
                    "input_kwargs": {},
                    "output": [
                        {
                            "type": "mindspore.Tensor",
                            "dtype": "Float32",
                            "shape": [
                                4096,
                                1,
                                1,
                                128
                            ],
                            "Max": 1.0000001192092896,
                            "Min": -1.0000001192092896,
                            "Mean": 0.13587358593940735,
                            "Norm": 528.9301147460938
                        }
                    ]
                },
                # Functional Type
                "Functional.flash_attention_score.0.forward": {
                    "input_args": [
                        {
                            "type": "mindspore.Tensor",
                            "dtype": "BFloat16",
                            "shape": [
                                4096,
                                1,
                                1536
                            ],
                            "Max": 3.671875,
                            "Min": -3.765625,
                            "Mean": -0.00072479248046875,
                            "Norm": 1744.0
                        },
                        {
                            "type": "mindspore.Tensor",
                            "dtype": "BFloat16",
                            "shape": [
                                4096,
                                1,
                                1536
                            ],
                            "Max": 3.484375,
                            "Min": -3.0625,
                            "Mean": -0.00115966796875,
                            "Norm": 1728.0
                        },
                        {
                            "type": "mindspore.Tensor",
                            "dtype": "BFloat16",
                            "shape": [
                                4096,
                                1,
                                1536
                            ],
                            "Max": 3.28125,
                            "Min": -3.234375,
                            "Mean": -0.001861572265625,
                            "Norm": 1744.0
                        },
                        {
                            "type": "int",
                            "value": 12
                        }
                    ],
                    "input_kwargs": {
                        "attn_mask": {
                            "type": "mindspore.Tensor",
                            "dtype": "UInt8",
                            "shape": [
                                1,
                                1,
                                4096,
                                4096
                            ],
                            "Max": 1.0,
                            "Min": 0.0,
                            "Mean": 0.876311182975769,
                            "Norm": 3834.326904296875
                        },
                        "scalar_value": {
                            "type": "float",
                            "value": 0.08838834764831843
                        },
                        "pre_tokens": {
                            "type": "int",
                            "value": 65536
                        },
                        "next_tokens": {
                            "type": "int",
                            "value": 0
                        },
                        "input_layout": {
                            "type": "str",
                            "value": "SBH"
                        }
                    },
                    "output": [
                        {
                            "type": "mindspore.Tensor",
                            "dtype": "BFloat16",
                            "shape": [
                                4096,
                                1,
                                1536
                            ],
                            "Max": 2.734375,
                            "Min": -2.578125,
                            "Mean": -0.001373291015625,
                            "Norm": 266.0
                        }
                    ]
                },
            }
        }

        self.ms_construct_source = {
            "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0": "Cell.network_with_loss.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
            "Mint.cos.0.forward": "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0",
            "Functional.flash_attention_score.0.forward": "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0"
        }
        self.ms_stack_source = {
            "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0": [
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 505, "
                "in _run_construct, \n output = self._run_forward_hook(inputs, output)",
                "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 201, "
                "in ParallelTransformerLayerForward, \n attention_output, _ = self.attention(",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, "
                "in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, "
                "in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative"
                "/transformer/transformer.py, line 1454, in construct, \n hidden_states = layer(",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, "
                "in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, "
                "in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative"
                "/transformer/language_model.py, line 579, in construct, \n encoder_output = self.encoder(encoder_input,",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, "
                "in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, "
                "in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/PanGu_ms/pangu/gpt_model.py, line 101, "
                "in construct, \n lm_output = self.language_model(tokens,",
            ],
            "Mint.cos.0.forward": [
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 505, "
                "in _run_construct, \n output = self._run_forward_hook(inputs, output)",
                "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, "
                "in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 98, "
                "in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /path_to_net/PanGu_ms/pangu/model/rotary_pos_embedding.py, line 123, "
                "in _apply_fused_rotary_pos_emb, \n cos_ = mint.cos(freqs).to(t.dtype)",
                "File /path_to_net/PanGu_ms/pangu/model/rotary_pos_embedding.py, line 136, "
                "in apply_rotary_pos_emb, \n return _apply_fused_rotary_pos_emb(t, freqs)",
                "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 619, "
                "in construct, \n query = apply_rotary_pos_emb(query, q_pos_emb, self.config)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, "
                "in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, "
                "in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 201, "
                "in ParallelTransformerLayerForward, \n attention_output, _ = self.attention(",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, "
                "in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, "
                "in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative"
                "/transformer/transformer.py, line 1454, in construct, \n hidden_states = layer(",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, "
                "in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, "
                "in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative"
                "/transformer/language_model.py, line 579, in construct, \n encoder_output = self.encoder(encoder_input,",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, "
                "in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, "
                "in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/PanGu_ms/pangu/gpt_model.py, line 101, "
                "in construct, \n lm_output = self.language_model(tokens,",
            ],
            "Functional.flash_attention_score.0.forward": [
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 505, in _run_construct, \n output = self._run_forward_hook(inputs, output)",
                "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 98, in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 637, in construct, \n output = ops.flash_attention_score(",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 201, in ParallelTransformerLayerForward, \n attention_output, _ = self.attention(",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/transformer.py, line 1454, in construct, \n hidden_states = layer(",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/language_model.py, line 579, in construct, \n encoder_output = self.encoder(encoder_input,",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
                "File /path_to_net/PanGu_ms/pangu/gpt_model.py, line 101, in construct, \n lm_output = self.language_model(tokens,",
            ]
        }

    def test_find_regard_scope(self):
        start_sign = "add"
        end_sign = "attention"
        start_pos, end_pos = find_regard_scope(
            self.lines, start_sign, end_sign)
        self.assertEqual(start_pos, 4)
        self.assertEqual(end_pos, 9)

    def test_find_stack_func_list(self):
        result = find_stack_func_list(self.lines)
        self.assertEqual(result, ([], ['attn_mask_add']))

    def test_get_dump_data_items_when_pt_valid_then_pass(self):
        result = get_dump_data_items(
            self.pt_dump, self.pt_stack, self.pt_construct, Const.PT_FRAMEWORK)
        expected_result = [
            {
                "data_name": "Functional.max_pool2d.0.forward",
                "construct_scope": 'Module.pool1.MaxPool2d.forward.0',
                "full_scope": "Module.pool1.max_pool2d"
            },
            {
                "data_name": "Functional.conv2d.1.forward",
                "construct_scope": "Module.conv2.Conv2d.forward.0",
                "full_scope": "Module.conv2.conv2d"
            },
            {
                "data_name": "Functional.linear.5.backward",
                "construct_scope": "Module.fc3.Linear.backward.1",
                "full_scope": "Module.fc3.linear"
            },
            {
                "data_name": "Functional.conv2d.2.forward",
                "construct_scope": None,
                "full_scope": "Module.conv2d"
            },
            {
                "data_name": "Module.conv1.Conv2d.backward.1",
                "construct_scope": None,
                "full_scope": "Module.conv1"
            }
        ]
        actual_values = [(res.data_name, res.construct_scope) for res in result]
        expect_values = [(item.get("data_name"), item.get("construct_scope")) for item in expected_result]
        self.assertListEqual(actual_values, expect_values)

    def test_get_dump_data_items_when_ms_functional_valid_then_pass(self):
        result = get_dump_data_items(
            self.ms_dump, self.ms_stack, self.ms_construct, Const.MS_FRAMEWORK)
        expected_result = [
            {
                "data_name": "Functional.add.0.forward",
                "construct_scope": "Cell.transformer_layers.0.attention.core_attention."
                "scale_mask_softmax.ScaleMaskSoftmax.forward.0",
                "full_scope": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.add"
            },
            {
                "data_name": "Functional.add.4.forward",
                "construct_scope": "Cell.transformer_layers.0.attention.core_attention."
                "scale_mask_softmax.ScaleMaskSoftmax.forward.0",
                "full_scope": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.add"
            },
            {
                "data_name": "Tensor.reshape.2.forward",
                "construct_scope": "Cell.transformer_layers.0.attention.ParallelAttention.forward.0",
                "full_scope": "Cell.transformer_layers.0.attention.reshape"
            }
        ]
        actual_values = [(res.data_name, res.construct_scope, res.full_scope) for res in result]
        expect_values = [(item.get("data_name"), item.get("construct_scope"), item.get("full_scope")) for item in expected_result]
        self.assertListEqual(actual_values, expect_values)

    def test_get_stack_in_lines_when_frame_func_then_pass(self):
        result = get_stack_in_lines(self.simplified_lines)
        expected_result = ["copy"]
        self.assertEqual(result, expected_result)

    def test_find_stack_func_list_when_frame_func_then_pass(self):
        result = find_stack_func_list(self.frame_func_lines)
        self.assertEqual(result, (["copy"], []))

    def test_get_dump_data_items_when_layer_and_mint_valid_then_pass(self):
        result = get_dump_data_items(self.ms_dump_source, self.ms_stack_source, self.ms_construct_source, Const.MS_FRAMEWORK)

        expected_result = [
            {
                "data_name": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.ParallelAttention.forward.0",
                "construct_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.ParallelTransformerLayer.forward.0",
                "full_scope": "Cell.network_with_loss.module.language_model.encoder.layers.0.attention"
            },
            {
                "data_name": "Mint.cos.0.forward",
                "construct_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.ParallelAttention.forward.0",
                "full_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.cos"
            },
            {
                "data_name": "Functional.flash_attention_score.0.forward",
                "construct_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.ParallelAttention.forward.0",
                "full_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.flash_attention_score"
            }
        ]
        # result store DumpDataItem Object List
        actual_values = [(res.data_name, res.construct_scope, res.full_scope) for res in result]
        expect_values = [(item.get("data_name"), item.get("construct_scope"), item.get("full_scope")) for item in expected_result]
        self.assertListEqual(actual_values, expect_values)

    # Test 2: stack and construct are empty
    def test_get_dump_data_items_when_empty_construct_stack_then_pass(self):
        dump = {
            "data": {
                "Functional.flash_attention_score.0.forward": {
                "input_args": [
                    {
                        "type": "mindspore.Tensor",
                        "dtype": "BFloat16",
                        "shape": [
                            4096,
                            1,
                            1536
                        ],
                        "Max": 3.671875,
                        "Min": -3.765625,
                        "Mean": -0.00072479248046875,
                        "Norm": 1744.0
                    },
                    {
                        "type": "mindspore.Tensor",
                        "dtype": "BFloat16",
                        "shape": [
                            4096,
                            1,
                            1536
                        ],
                        "Max": 3.484375,
                        "Min": -3.0625,
                        "Mean": -0.00115966796875,
                        "Norm": 1728.0
                    },
                    {
                        "type": "mindspore.Tensor",
                        "dtype": "BFloat16",
                        "shape": [
                            4096,
                            1,
                            1536
                        ],
                        "Max": 3.28125,
                        "Min": -3.234375,
                        "Mean": -0.001861572265625,
                        "Norm": 1744.0
                    },
                    {
                        "type": "int",
                        "value": 12
                    }
                ],
                }
            }
        }
        stack = {}
        construct = {}
        framework = Const.MS_FRAMEWORK

        result = get_dump_data_items(dump, stack, construct, framework)

        self.assertEqual(result, [])

    # Test 3: No data in dump
    def test_get_dump_data_items_when_empty_dump_stack_then_pass(self):
        dump = {}
        stack = {}
        construct = {
            "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0": "Cell.network_with_loss.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0"
        }
        framework = Const.MS_FRAMEWORK

        result = get_dump_data_items(dump, stack, construct, framework)

        self.assertEqual(result, [])

    # Test 4: output_path is provided, ensure file is saved
    def test_empty_dump_with_output_path(self):
        dump = {}
        stack = {
            "Tensor.__add__.0.forward": ["stack data"]
        }
        construct = {
            "Tensor.__add__.0.forward": "construct data"
        }
        framework = Const.MS_FRAMEWORK
        output_path = "./"

        result = get_dump_data_items(dump, stack, construct, framework, output_path)

        # Check if file was saved
        entries = os.listdir(output_path)
        # filter file endwith _data.yaml
        sign = f"{Const.MS_FRAMEWORK}_data"
        data_yaml_files = [os.path.join(output_path, entry) for entry in entries if sign in entry]
        saved_file = data_yaml_files[0]
        yaml_info = load_yaml(saved_file)
        os.remove(saved_file)
        self.assertEqual(yaml_info, {})

    # Test 5: Empty dump and output_path
    def test_get_dump_data_items_when_valid_with_output_path_then_pass(self):
        output_path = "./"
        result = get_dump_data_items(self.ms_dump_source, self.ms_stack_source, self.ms_construct_source, Const.MS_FRAMEWORK, output_path)
        # Check if file was saved
        entries = os.listdir(output_path)
        sign = f"{Const.MS_FRAMEWORK}_data"
        data_yaml_files = [os.path.join(output_path, entry) for entry in entries if sign in entry]
        # filter file endwith _data.yaml
        saved_file = data_yaml_files[0]
        yaml_info = load_yaml(saved_file)
        expected_result = [
            {
                "data_name": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.ParallelAttention.forward.0",
                "construct_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.ParallelTransformerLayer.forward.0",
                "full_scope": "Cell.network_with_loss.module.language_model.encoder.layers.0.attention"
            },
            {
                "data_name": "Mint.cos.0.forward",
                "construct_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.ParallelAttention.forward.0",
                "full_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.cos"
            },
            {
                "data_name": "Functional.flash_attention_score.0.forward",
                "construct_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.ParallelAttention.forward.0",
                "full_scope": "Cell.network_with_loss.module.language_model.encoder.layers."
                "0.attention.flash_attention_score"
            }
        ]
        actual_values = [(name, res.get("construct_scope"), res.get("full_scope")) for name, res in yaml_info.items()]
        expect_values = [(item.get("data_name"), item.get("construct_scope"), item.get("full_scope")) for item in expected_result]
        os.remove(saved_file)
        self.assertListEqual(actual_values, expect_values)
