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

import unittest
from pathlib import Path

from msprobe.core.compare.layer_mapping import generate_api_mapping_by_layer_mapping, \
    generate_data_mapping_by_layer_mapping


class TestLayerMapping(unittest.TestCase):
    def setUp(self):
        # 获取当前文件路径
        self.current_dir = Path(__file__).parent
        self.npu_dump_json = self.current_dir / "dump_file/mindspore_data/dump.json"
        self.bench_dump_json = self.current_dir / "dump_file/pytorch_data/dump.json"
        self.layer_mapping = str(self.current_dir / "dump_file/layer_mapping.yaml")

    def test_generate_api_mapping_by_layer_mapping_then_pass(self):
        # Example test to check if construct.json is processed correctly
        res = generate_api_mapping_by_layer_mapping(self.npu_dump_json, self.bench_dump_json, self.layer_mapping)
        excepted_api_mapping = {
            "Tensor.__add__.0.forward": "N/A",
            "Tensor.__bool__.1.forward": "N/A",
            "Tensor.__add__.1.forward": "Tensor.__add__.0.forward",
            "Mint.logical_or.0.forward": "Tensor.__or__.0.forward",
            "Distributed.all_reduce.0.forward": "Distributed.all_reduce.0.forward",
            "Cell.network_with_loss.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0": "Module.module.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0",
            "Primitive.norm.RmsNorm.0.forward": "NPU.npu_rms_norm.0.forward",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0": "Module.module.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.input_norm.FusedRMSNorm.backward.0": "N/A",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0": "Module.module.module.language_model.encoder.layers.0.self_attention.ParallelAttention.forward.0",
            "Mint.cos.0.forward": "Torch.cos.0.forward",
            "Functional.flash_attention_score.0.forward": "NPU.npu_fusion_attention.0.forward",
            "Functional.flash_attention_score.0.backward": "NPU.npu_fusion_attention.0.backward",
        }
        self.assertDictEqual(res, excepted_api_mapping)

    def test_generate_data_mapping_by_layer_mapping_then_pass(self):
        input_param = {"npu_path": self.npu_dump_json, "bench_path": self.bench_dump_json}
        res = generate_data_mapping_by_layer_mapping(input_param, self.layer_mapping)
        excepted_data_mapping = {
            "Tensor.__add__.0.forward.input.0": "N/A",
            "Tensor.__add__.0.forward.input.1": "N/A",
            "Tensor.__add__.0.forward.output.0": "N/A",
            "Tensor.__bool__.1.forward.input.0": "N/A",
            "Tensor.__bool__.1.forward.output.0": "N/A",
            "Tensor.__add__.1.forward.input.0": "Tensor.__add__.0.forward.input.0",
            "Tensor.__add__.1.forward.input.1": "Tensor.__add__.0.forward.input.1",
            "Tensor.__add__.1.forward.output.0": "Tensor.__add__.0.forward.output.0",
            "Mint.logical_or.0.forward.input.0": "Tensor.__or__.0.forward.input.0",
            "Mint.logical_or.0.forward.input.1": "Tensor.__or__.0.forward.input.1",
            "Mint.logical_or.0.forward.output.0": "Tensor.__or__.0.forward.output.0",
            "Distributed.all_reduce.0.forward.input.0": "Distributed.all_reduce.0.forward.input.0",
            "Distributed.all_reduce.0.forward.input.group": "Distributed.all_reduce.0.forward.input.group",
            "Distributed.all_reduce.0.forward.output.0": "Distributed.all_reduce.0.forward.output.0",
            "Distributed.all_reduce.0.forward.output.1": "Distributed.all_reduce.0.forward.output.1",
            "Cell.network_with_loss.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0.input.0": "Module.module.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0.input.0",
            "Cell.network_with_loss.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0.output.0": "Module.module.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0.output.0",
            "Primitive.norm.RmsNorm.0.forward.input.0": "NPU.npu_rms_norm.0.forward.input.0",
            "Primitive.norm.RmsNorm.0.forward.input.1": "NPU.npu_rms_norm.0.forward.input.1",
            "Primitive.norm.RmsNorm.0.forward.output.0": "NPU.npu_rms_norm.0.forward.output.0",
            "Primitive.norm.RmsNorm.0.forward.output.1": "NPU.npu_rms_norm.0.forward.output.1",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0.input.0": "Module.module.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0.input.0",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0.output.0": "Module.module.module.language_model.encoder.layers.0.ParallelTransformerLayer.forward.0.output.0",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.input_norm.FusedRMSNorm.backward.0.input.0": "N/A",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.input_norm.FusedRMSNorm.backward.0.output.0": "N/A",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0.output.0": "Module.module.module.language_model.encoder.layers.0.self_attention.ParallelAttention.forward.0.output.0",
            "Cell.network_with_loss.module.language_model.encoder.layers.0.attention.ParallelAttention.forward.0.output.1": "Module.module.module.language_model.encoder.layers.0.self_attention.ParallelAttention.forward.0.output.1",
            "Mint.cos.0.forward.input.0": "Torch.cos.0.forward.input.0",
            "Mint.cos.0.forward.output.0": "Torch.cos.0.forward.output.0",
            "Functional.flash_attention_score.0.forward.input.0": "NPU.npu_fusion_attention.0.forward.input.0",
            "Functional.flash_attention_score.0.forward.input.1": "NPU.npu_fusion_attention.0.forward.input.1",
            "Functional.flash_attention_score.0.forward.input.2": "NPU.npu_fusion_attention.0.forward.input.2",
            "Functional.flash_attention_score.0.forward.input.3": "NPU.npu_fusion_attention.0.forward.input.3",
            "Functional.flash_attention_score.0.forward.input.attn_mask": "N/A",
            "Functional.flash_attention_score.0.forward.input.scalar_value": "N/A",
            "Functional.flash_attention_score.0.forward.input.pre_tokens": "N/A",
            "Functional.flash_attention_score.0.forward.input.next_tokens": "N/A",
            "Functional.flash_attention_score.0.forward.input.input_layout": "N/A",
            "Functional.flash_attention_score.0.forward.output.0": "NPU.npu_fusion_attention.0.forward.output.0",
            "Functional.flash_attention_score.0.backward.input.0": "NPU.npu_fusion_attention.0.backward.input.0",
            "Functional.flash_attention_score.0.backward.output.0": "NPU.npu_fusion_attention.0.backward.output.0",
            "Functional.flash_attention_score.0.backward.output.1": "NPU.npu_fusion_attention.0.backward.output.1",
            "Functional.flash_attention_score.0.backward.output.2": "NPU.npu_fusion_attention.0.backward.output.2",
            "Functional.flash_attention_score.0.backward.output.3": "NPU.npu_fusion_attention.0.backward.output.3",
        }
        self.assertDictEqual(res, excepted_data_mapping)
