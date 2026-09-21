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

import pytest
import unittest


from msprobe.core.common.utils import CompareException
from msprobe.core.compare.layer_mapping.data_scope_parser import (
    DumpDataItem,
)
from msprobe.core.compare.layer_mapping.postprocess_pass import (
    postprocess_pass,
    backward_pass,
    renumber_index_pass,
)
from msprobe.core.common.const import Const


class TestModifyMapping(unittest.TestCase):
    def setUp(self):
        pt_name1 = "Distributed.all_reduce.0.forward"
        pt_construct_info1 = "Module.module.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0"
        pt_stack_info1 = [
            "File /path_to_package/mstt/debug/accuracy_tools/msprobe/pytorch/hook_module/wrap_distributed.py, line 68, in distributed_op_template, \n return DistributedOPTemplate(op_name, hook)(*args, **kwargs)",
            "File /path_to_net/third_party/Megatron-LM/megatron/core/tensor_parallel/mappings.py, line 24, in _reduce, \n torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())",
            "File /path_to_net/third_party/Megatron-LM/megatron/core/tensor_parallel/mappings.py, line 223, in forward, \n return _reduce(input_)",
            "File /path_to_package/site-packages/torch/autograd/function.py, line 539, in apply, \n return super().apply(*args, **kwargs)  # type: ignore[misc]",
            "File /path_to_net/third_party/Megatron-LM/megatron/core/tensor_parallel/mappings.py, line 436, in reduce_from_tensor_model_parallel_region, \n return _ReduceFromModelParallelRegion.apply(input_)",
            "File /path_to_package/third_party/MindSpeed/mindspeed/core/tensor_parallel/layers.py, line 35, in vocab_parallel_embedding_forward, \n output = reduce_from_tensor_model_parallel_region(output_parallel)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/language_model.py, line 217, in forward, \n words_embeddings = self.word_embeddings(input_ids)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/language_model.py, line 473, in forward, \n encoder_input = self.embedding(enc_input_ids, enc_position_ids,",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/gpt_model.py, line 86, in forward, \n lm_output = self.language_model(",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/module.py, line 190, in forward, \n outputs = self.module(*inputs, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /path_to_net/third_party/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py, line 179, in forward, \n return self.module(*inputs, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
            "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /path_to_net/PanGu/pretrain_gpt.py, line 247, in forward_step, \n output_tensor = model(tokens, position_ids, attention_mask,",
            "File /path_to_net/third_party/Megatron-LM/megatron/core/pipeline_parallel/schedules.py, line 193, in forward_step, \n output_tensor, loss_func = forward_step_func(data_iterator, model)",
            "File /path_to_net/third_party/Megatron-LM/megatron/core/pipeline_parallel/schedules.py, line 1225, in forward_backward_pipelining_without_interleaving, \n output_tensor = forward_step(",
            "File /path_to_net/third_party/Megatron-LM/megatron/training/training.py, line 624, in train_step, \n losses_reduced = forward_backward_func(",
            "File /path_to_net/PanGu/pangu/training/auto_parallel_wrapper.py, line 34, in wrapper, \n ret = train_step(*args, **kwargs)",
            "File /path_to_net/PanGu/pangu/training/training.py, line 495, in train, \n train_step(forward_step_func,",
            "File /path_to_net/PanGu/pangu/training/training.py, line 303, in pretrain, \n iteration, num_floating_point_operations_so_far = train(",
            "File /path_to_net/PanGu/pretrain_gpt.py, line 372, in main, \n pretrain(train_valid_test_datasets_provider,",
            "File /path_to_net/PanGu/pretrain_gpt.py, line 392, in <module>, \n main()"
        ]
        pt_item1 = DumpDataItem(Const.PT_FRAMEWORK)
        pt_item1.set(pt_name1, pt_construct_info1, pt_stack_info1)

        pt_name2 = "Module.module.module.language_model.encoder.layers.0.self_attention.ParallelAttention.forward.0"
        pt_construct_info2 = "Module.module.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0"
        pt_stack_info2 = [
        "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/transformer.py, line 1198, in forward, \n self.self_attention(",
        "File /path_to_package/third_party/MindSpeed/mindspeed/core/transformer/transformer.py, line 21, in row_parallel_forward, \n output = forward_func(*args, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
        "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/transformer.py, line 1832, in forward, \n hidden_states = layer(",
        "File /path_to_package/third_party/MindSpeed/mindspeed/model/transformer.py, line 349, in wrapper, \n return fn(self, hidden_states, attention_mask, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
        "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/language_model.py, line 500, in forward, \n encoder_output = self.encoder(",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
        "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/gpt_model.py, line 86, in forward, \n lm_output = self.language_model(",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
        "File /path_to_net/third_party/Megatron-LM/megatron/legacy/model/module.py, line 190, in forward, \n outputs = self.module(*inputs, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
        "File /path_to_net/third_party/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py, line 179, in forward, \n return self.module(*inputs, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)",
        "File /path_to_package/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
        "File /path_to_net/PanGu/pretrain_gpt.py, line 247, in forward_step, \n output_tensor = model(tokens, position_ids, attention_mask,",
        "File /path_to_net/third_party/Megatron-LM/megatron/core/pipeline_parallel/schedules.py, line 193, in forward_step, \n output_tensor, loss_func = forward_step_func(data_iterator, model)",
        "File /path_to_net/third_party/Megatron-LM/megatron/core/pipeline_parallel/schedules.py, line 1225, in forward_backward_pipelining_without_interleaving, \n output_tensor = forward_step(",
        "File /path_to_net/third_party/Megatron-LM/megatron/training/training.py, line 624, in train_step, \n losses_reduced = forward_backward_func(",
        "File /path_to_net/PanGu/pangu/training/auto_parallel_wrapper.py, line 34, in wrapper, \n ret = train_step(*args, **kwargs)",
        "File /path_to_net/PanGu/pangu/training/training.py, line 495, in train, \n train_step(forward_step_func,",
        "File /path_to_net/PanGu/pangu/training/training.py, line 303, in pretrain, \n iteration, num_floating_point_operations_so_far = train(",
        "File /path_to_net/PanGu/pretrain_gpt.py, line 372, in main, \n pretrain(train_valid_test_datasets_provider,",
        "File /path_to_net/PanGu/pretrain_gpt.py, line 392, in <module>, \n main()"
        ]
        pt_item2 = DumpDataItem(Const.PT_FRAMEWORK)
        pt_item2.set(pt_name2, pt_construct_info2, pt_stack_info2)

        """
        ----------------------------------------------------------
        Normal Case Data
        ----------------------------------------------------------
        """
        ms_name1 = "Distributed.all_reduce.0.forward"
        ms_construct_info1 = "Cell.network_with_loss.module.language_model.embedding.word_embeddings.reduce_from_mp_region.ReduceFromModelParallelRegion.forward.0"
        ms_stack_info1 = [
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 505, in _run_construct, \n output = self._run_forward_hook(inputs, output)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
            "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 98, in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/tensor_parallel/mappings.py, line 241, in construct, \n output = comm_func.all_reduce(input_, group=self.tp_group)[0]",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 785, in _call_custom_bprop, \n output = self.construct(*args, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2450, in _backward_hook_construct, \n outputs = self._call_custom_bprop(outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/tensor_parallel/layers.py, line 1168, in construct, \n output = self.reduce_from_mp_region(output_parallel)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2455, in _backward_hook_construct, \n outputs = self.construct(outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/language_model.py, line 226, in construct, \n words_embeddings = self.word_embeddings(input_ids)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/language_model.py, line 554, in construct, \n text_embedding_out = self.embedding(enc_input_ids, enc_position_ids,",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/PanGu_ms/pangu/gpt_model.py, line 101, in construct, \n lm_output = self.language_model(tokens,",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/distributed/distributed_data_parallel.py, line 171, in construct, \n output = self.module(*inputs, **inputs_dict)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 747, in _complex_call, \n output = self._run_construct(*args, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 735, in __call__, \n return self._complex_call(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/pipeline_parallel/schedules.py, line 113, in run_forward, \n output_tensor = model(*input_data)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/pipeline_parallel/schedules.py, line 760, in forward_backward_pipelining_without_interleaving, \n micro_input_data = run_forward(*micro_input_data,",
            "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 444, in forward_backward_with_pipelining, \n loss, logits, grads = forward_backward_pipelining_without_interleaving(",
            "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 580, in construct, \n (loss, _), grads = self.forward_backward_func(*inputs_tuple, loss_scale=current_step_loss_scale, **inputs_dict)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 731, in __call__, \n return self.construct(*args, **kwargs)",
            "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 725, in train, \n loss, is_finite, loss_scale, learning_rate, _ = train_one_step_cell(**data)",
            "File /path_to_net/PanGu_ms/pretrain_gpt.py, line 329, in main, \n train(",
            "File /path_to_net/PanGu_ms/pretrain_gpt.py, line 342, in <module>, \n main()"
        ]
        ms_item1 = DumpDataItem(Const.MS_FRAMEWORK)
        ms_item1.set(ms_name1, ms_construct_info1, ms_stack_info1)
        ms_item1.type_name = "all_reduce"
        ms_item1.layer_scope = "layer_2"
        ms_item1.full_scope = "Cell.network_with_loss.module.language_model.embedding.word_embeddings.reduce_from_mp_region.ReduceFromModelParallelRegion.all_reduce.0"
        
        """
        ----------------------------------------------------------
        Used for renumber layer id
        ----------------------------------------------------------
        """
        ms_name2 = "Cell.network_with_loss.module.language_model.encoder.layers.1.attention.ParallelAttention.forward.0"
        ms_construct_info2 = "Cell.network_with_loss.module.language_model.encoder.layers.1.ParallelTransformerLayer.forward.0"
        ms_stack_info2 = [
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 505, in _run_construct, \n output = self._run_forward_hook(inputs, output)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 201, in ParallelTransformerLayerForward, \n attention_output, _ = self.attention(",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/transformer.py, line 1454, in construct, \n hidden_states = layer(",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/language_model.py, line 579, in construct, \n encoder_output = self.encoder(encoder_input,",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/PanGu_ms/pangu/gpt_model.py, line 101, in construct, \n lm_output = self.language_model(tokens,",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/distributed/distributed_data_parallel.py, line 171, in construct, \n output = self.module(*inputs, **inputs_dict)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 747, in _complex_call, \n output = self._run_construct(*args, **kwargs)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 735, in __call__, \n return self._complex_call(*args, **kwargs)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/pipeline_parallel/schedules.py, line 113, in run_forward, \n output_tensor = model(*input_data)",
            "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/pipeline_parallel/schedules.py, line 760, in forward_backward_pipelining_without_interleaving, \n micro_input_data = run_forward(*micro_input_data,",
            "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 444, in forward_backward_with_pipelining, \n loss, logits, grads = forward_backward_pipelining_without_interleaving(",
            "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 580, in construct, \n (loss, _), grads = self.forward_backward_func(*inputs_tuple, loss_scale=current_step_loss_scale, **inputs_dict)",
            "File /path_to_package/site-packages/mindspore/nn/cell.py, line 731, in __call__, \n return self.construct(*args, **kwargs)",
            "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 725, in train, \n loss, is_finite, loss_scale, learning_rate, _ = train_one_step_cell(**data)",
            "File /path_to_net/PanGu_ms/pretrain_gpt.py, line 329, in main, \n train(",
            "File /path_to_net/PanGu_ms/pretrain_gpt.py, line 342, in <module>, \n main()"
        ]
        ms_item2 = DumpDataItem(Const.MS_FRAMEWORK)
        ms_item2.set(ms_name2, ms_construct_info2, ms_stack_info2)
    
        """
        ----------------------------------------------------------
        backward sample data used for backward_pass
        ----------------------------------------------------------
        """
        ms_name3 = "Functional.flash_attention_score.0.backward"
        ms_construct_info3 = "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.backward.0"
        ms_stack_info3 = []
        ms_item3 = DumpDataItem(Const.MS_FRAMEWORK)
        ms_item3.set(ms_name3, ms_construct_info3, ms_stack_info3)


        """
        ----------------------------------------------------------
        corresponding forward for backward sample data used for backward_pass
        ----------------------------------------------------------
        """
        ms_name4 = "Functional.flash_attention_score.0.forward"
        ms_construct_info4 = "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0"
        ms_stack_info4 = [
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 505, in _run_construct, \n output = self._run_forward_hook(inputs, output)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
        "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
        "File /path_to_package/mstt/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 98, in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
        "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 637, in construct, \n output = ops.flash_attention_score(",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
        "File /path_to_net/PanGu_ms/pangu/model/transformer.py, line 201, in ParallelTransformerLayerForward, \n attention_output, _ = self.attention(",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
        "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/transformer.py, line 1454, in construct, \n hidden_states = layer(",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
        "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/transformer/language_model.py, line 579, in construct, \n encoder_output = self.encoder(encoder_input,",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
        "File /path_to_net/PanGu_ms/pangu/gpt_model.py, line 101, in construct, \n lm_output = self.language_model(tokens,",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 733, in __call__, \n return self._run_construct(*args, **kwargs)",
        "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/distributed/distributed_data_parallel.py, line 171, in construct, \n output = self.module(*inputs, **inputs_dict)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 2453, in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 494, in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 747, in _complex_call, \n output = self._run_construct(*args, **kwargs)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 735, in __call__, \n return self._complex_call(*args, **kwargs)",
        "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/pipeline_parallel/schedules.py, line 113, in run_forward, \n output_tensor = model(*input_data)",
        "File /path_to_net/third_party/dynamic-parallel/mindformers/experimental/parallel_core/pynative/pipeline_parallel/schedules.py, line 760, in forward_backward_pipelining_without_interleaving, \n micro_input_data = run_forward(*micro_input_data,",
        "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 444, in forward_backward_with_pipelining, \n loss, logits, grads = forward_backward_pipelining_without_interleaving(",
        "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 580, in construct, \n (loss, _), grads = self.forward_backward_func(*inputs_tuple, loss_scale=current_step_loss_scale, **inputs_dict)",
        "File /path_to_package/site-packages/mindspore/nn/cell.py, line 731, in __call__, \n return self.construct(*args, **kwargs)",
        "File /path_to_net/PanGu_ms/pangu/pynative/training/training.py, line 725, in train, \n loss, is_finite, loss_scale, learning_rate, _ = train_one_step_cell(**data)",
        "File /path_to_net/PanGu_ms/pretrain_gpt.py, line 329, in main, \n train(",
        "File /path_to_net/PanGu_ms/pretrain_gpt.py, line 342, in <module>, \n main()"
        ]
        ms_item4 = DumpDataItem(Const.MS_FRAMEWORK)
        ms_item4.set(ms_name4, ms_construct_info4, ms_stack_info4)
        self.ms_data_items = [ms_item1, ms_item2, ms_item3, ms_item4]
        self.pt_data_items = [pt_item1, pt_item2]

    def test_backward_pass_when_ms_valid_then_pass(self):
        name2item = {data_item.data_name : data_item for data_item in self.ms_data_items}
        backward_pass(self.ms_data_items, name2item)
        expected_stack_scope = self.ms_data_items[3].stack_scope

        self.assertEqual(self.ms_data_items[2].stack_scope, self.ms_data_items[3].stack_scope)
        self.assertEqual(self.ms_data_items[2].full_scope, self.ms_data_items[3].full_scope)
        self.assertEqual(self.ms_data_items[2].layer_scope, self.ms_data_items[3].layer_scope)

    def test_backward_pass_when_none_then_pass(self):
        with self.assertRaises(CompareException) as context:
            non_data = DumpDataItem(Const.MS_FRAMEWORK)
            non_data.set('', '', [])
            non_datas = [non_data, non_data]
            name2item = {data_item.data_name : data_item for data_item in non_datas}
            backward_pass(non_datas, name2item)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)

    def test_renumber_index_pass_when_ms_valid_then_pass(self):
        suffix = "layers"
        type_name = "ParallelTransformer"
        renumber_index_pass(self.ms_data_items, type_name, suffix)
        self.assertEqual(self.ms_data_items[1].full_scope, "Cell.network_with_loss.module.language_model.encoder.layers.1.attention")

    def test_postprocess_pass_when_ms_valid_then_pass(self):
        name2item = {data_item.data_name : data_item for data_item in self.ms_data_items}
        try:
            postprocess_pass(self.ms_data_items, name2item)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_postprocess_pass_when_pt_valid_then_pass(self):
        name2item = {data_item.data_name : data_item for data_item in self.pt_data_items}
        try:
            postprocess_pass(self.pt_data_items, name2item)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")       

    def test_postprocess_pass_when_non_data_then_pass(self):
        with self.assertRaises(CompareException) as context:
            non_data = DumpDataItem(Const.MS_FRAMEWORK)
            non_data.set('', '', [])
            non_datas = [non_data, non_data]
            name2item = {data_item.data_name : data_item for data_item in non_datas}
            postprocess_pass(non_datas, name2item)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)