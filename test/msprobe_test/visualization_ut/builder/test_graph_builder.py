import os
import unittest
from unittest.mock import MagicMock, patch
from msprobe.visualization.builder.graph_builder import GraphBuilder, Graph, GraphExportConfig
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.graph.base_node import BaseNode


class TestGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.construct_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "construct.json")
        self.construct_path_empty = os.path.join(os.path.dirname(os.path.realpath(__file__)), "construct_empty.json")
        self.data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dump.json")
        self.stack_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "stack.json")
        self.model_name = "TestModel"
        self.graph = Graph(self.model_name)
        self.graph_b = Graph(self.model_name)
        self.config = GraphExportConfig(self.graph, self.graph_b)
        self.construct_dict = {
            "Tensor1": "Module1",
            "Module1": None
        }
        self.data_dict = {
            "Module1": {"data": "data for Module1"},
            "Tensor1": {"data": "data for Tensor1"}
        }
        self.stack_dict = {}

    def test_build(self):
        graph = GraphBuilder.build(self.construct_path, self.data_path, self.stack_path, self.model_name)
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, Graph)
        self.assertEqual(len(graph.node_map), 3)

        with self.assertRaises(RuntimeError):
            GraphBuilder.build(self.construct_path_empty, self.data_path, self.stack_path, self.model_name)

    @patch('msprobe.visualization.graph.node_op.NodeOp.get_node_op')
    @patch('msprobe.visualization.builder.msprobe_adapter.get_input_output', return_value=([], []))
    def test__init_nodes(self, mock_get_input_output, mock_get_node_op):
        GraphBuilder._init_nodes(self.graph, self.construct_dict, self.data_dict, self.stack_dict)
        mock_get_node_op.assert_any_call("Tensor1")
        mock_get_node_op.assert_any_call("Module1")
        self.assertIs(self.graph.root, self.graph.get_node("TestModel"))

    def test__create_or_get_node(self):
        node_op = MagicMock()
        data_dict = {"node1": {}}
        stack_dict = {}
        node = GraphBuilder._create_or_get_node(self.graph, [data_dict, stack_dict], node_op, "node1")
        self.assertIn("node1", self.graph.node_map)
        self.assertEqual(node.input_data, {})
        self.assertEqual(node.output_data, {})

    def test__handle_backward_upnode_missing(self):
        construct_dict = {'Module.module.a.forward.0': 'Module.root.forward.0', 'Module.module.a.backward.0': None,
                          'Module.root.forward.0': None, 'Module.root.backward.0': None,
                          'Module.module.b.forward.0': 'Module.root.forward.0',
                          'Module.module.b.backward.0': 'Module.root.backward.0', 'Module.module.c.backward.0': None}
        node_id_a = GraphBuilder._handle_backward_upnode_missing(construct_dict, 'Module.module.a.backward.0', None)
        self.assertEqual(node_id_a, 'Module.root.backward.0')
        node_id_b = GraphBuilder._handle_backward_upnode_missing(construct_dict, 'Module.module.b.backward.0',
                                                                 'Module.root.backward.0')
        self.assertEqual(node_id_b, 'Module.root.backward.0')
        node_id_c = GraphBuilder._handle_backward_upnode_missing(construct_dict, 'Module.module.c.backward.0', None)
        self.assertIsNone(node_id_c)
        construct_dict = {'Module.module.a.forward': 'Module.root.forward', 'Module.module.a.backward': None,
                          'Module.root.forward': None, 'Module.root.backward': None,
                          'Module.module.b.forward': 'Module.root.forward',
                          'Module.module.b.backward': 'Module.root.backward', 'Module.module.c.backward': None}
        node_id_a = GraphBuilder._handle_backward_upnode_missing(construct_dict, 'Module.module.a.backward', None)
        self.assertEqual(node_id_a, 'Module.root.backward')
        node_id_b = GraphBuilder._handle_backward_upnode_missing(construct_dict, 'Module.module.b.backward',
                                                                 'Module.root.backward')
        self.assertEqual(node_id_b, 'Module.root.backward')
        node_id_c = GraphBuilder._handle_backward_upnode_missing(construct_dict, 'Module.module.c.backward', None)
        self.assertIsNone(node_id_c)

    def test__collect_apis_between_modules_only_apis(self):
        graph = Graph('TestNet')
        graph.root.subnodes = [BaseNode(NodeOp.function_api, 'Tensor.a.0'), BaseNode(NodeOp.function_api, 'Tensor.b.0')]
        GraphBuilder._collect_apis_between_modules(graph)
        self.assertEqual(len(graph.root.subnodes), 1)
        self.assertEqual(graph.root.subnodes[0].op, NodeOp.api_collection)
        self.assertEqual(len(graph.root.subnodes[0].subnodes), 2)
        self.assertEqual(graph.root.subnodes[0].id, 'Apis_Between_Modules.0')

    def test__collect_apis_between_modules_mixed_nodes(self):
        graph = Graph('TestNet')
        graph.root.subnodes = [BaseNode(NodeOp.function_api, 'Tensor.a.0'), BaseNode(NodeOp.module, 'Module.a.0'),
                               BaseNode(NodeOp.module, 'Module.b.0'), BaseNode(NodeOp.function_api, 'Tensor.b.0'),
                               BaseNode(NodeOp.function_api, 'Tensor.c.0'), BaseNode(NodeOp.module, 'Module.a.1')]
        GraphBuilder._collect_apis_between_modules(graph)
        self.assertEqual(len(graph.root.subnodes), 5)
        self.assertEqual(graph.root.subnodes[0].op, NodeOp.function_api)
        self.assertEqual(graph.root.subnodes[1].op, NodeOp.module)
        self.assertEqual(graph.root.subnodes[3].op, NodeOp.api_collection)
        self.assertEqual(len(graph.root.subnodes[3].subnodes), 2)
        self.assertEqual(graph.root.subnodes[3].id, 'Apis_Between_Modules.0')

    def test__collect_apis_between_modules_only_modules(self):
        graph = Graph('TestNet')
        graph.root.subnodes = [BaseNode(NodeOp.module, 'Module.a.0'), BaseNode(NodeOp.module, 'Module.b.0'),
                               BaseNode(NodeOp.module, 'Module.a.1')]
        GraphBuilder._collect_apis_between_modules(graph)
        self.assertEqual(len(graph.root.subnodes), 3)
        self.assertEqual(graph.root.subnodes[0].op, NodeOp.module)
        self.assertEqual(graph.root.subnodes[1].op, NodeOp.module)
        self.assertEqual(graph.root.subnodes[2].op, NodeOp.module)
        self.assertEqual(len(graph.root.subnodes[0].subnodes), 0)
        self.assertEqual(graph.root.subnodes[0].id, 'Module.a.0')

    def test_add_parameters_grad(self):
        graph = Graph('TestNet')
        graph.add_node(NodeOp.module, 'Module.a.backward.0', graph.root)
        graph.add_node(NodeOp.module, 'Module.b.backward.0', graph.root)
        graph.add_node(NodeOp.module, 'Module.a.backward.1', graph.root)
        graph.add_node(NodeOp.module, 'Module.aa.backward.0', graph.get_node('Module.a.backward.0'))
        graph.add_node(NodeOp.module, 'Module.aaa.backward.0', graph.get_node('Module.a.backward.0'))
        graph.add_node(NodeOp.module, 'Module.aa.backward.1', graph.get_node('Module.a.backward.1'))
        graph.add_node(NodeOp.module, 'Module.aaa.backward.1', graph.get_node('Module.a.backward.1'))

        data_dict = {'Module.a.parameters_grad': {}, 'Module.aaa.parameters_grad': {}}
        GraphBuilder._add_parameters_grad(graph, data_dict)
        root_nodes_id = [node.id for node in graph.get_node('TestNet').subnodes]
        sub_nodes_id0 = [node.id for node in graph.get_node('Module.a.backward.0').subnodes]
        sub_nodes_id1 = [node.id for node in graph.get_node('Module.a.backward.1').subnodes]

        self.assertEqual(root_nodes_id[-1], 'Module.a.backward.1')
        self.assertEqual(sub_nodes_id0[-1], 'Module.aaa.backward.0')
        self.assertEqual(sub_nodes_id1[-1], 'Module.a.parameters_grad')

    def test_handle_backward_inplace(self):
        construct_dict = {'Module.module.Float16Model.forward.0': None,
                          'Module.module.layer1.BasicBlock.forward.0': 'Module.module.Float16Model.forward.0',
                          'Module.module.layer2.BasicBlock.forward.0': 'Module.module.Float16Model.forward.0',
                          'Module.module.conv.Conv2d.forward.0': 'Module.module.Float16Model.forward.0',
                          'Module.module.Float16Model.backward.0': None,
                          'Module.module.layer1.BasicBlock.backward.0': 'Module.module.Float16Model.backward.0',
                          'Module.module.layer2.BasicBlock.backward.0': 'Module.module.conv.Conv2d.backward.0',
                          'Module.module.conv.Conv2d.backward.0': 'Module.module.Float16Model.backward.0'
                          }
        up_node_id = GraphBuilder._handle_backward_inplace(construct_dict,
                                                           'Module.module.layer2.BasicBlock.backward.0',
                                                           'Module.module.conv.Conv2d.backward.0')
        self.assertEqual(up_node_id, 'Module.module.Float16Model.backward.0')
        up_node_id = GraphBuilder._handle_backward_inplace(construct_dict,
                                                           'Module.module.layer1.BasicBlock.backward.0',
                                                           'Module.module.Float16Model.backward.0')
        self.assertEqual(up_node_id, 'Module.module.Float16Model.backward.0')

    def test_is_valid_batch_p2p_output(self):
        self.assertFalse(GraphBuilder._is_valid_batch_p2p_output('a'))
        self.assertFalse(GraphBuilder._is_valid_batch_p2p_output([]))
        self.assertTrue(GraphBuilder._is_valid_batch_p2p_output([['a']]))

    def test_extract_batch_p2p_info(self):
        node_data = {
            "output": [[{'a': 1}], [{'b': 1}]]
        }
        GraphBuilder._extract_batch_p2p_info(self.graph.root, node_data)
        self.assertEqual(self.graph.root.batch_p2p_info, [{'group_id': None, 'op': None, 'peer': None}])

    def test_is_recompute_by_stack_torch(self):
        stack_list = [
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1829, "
            "in inner, \n result = forward_call(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1881, "
            "in _call_impl, \n return inner()",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1775, "
            "in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /root/work/filestorage/gh/code/MOVA-feat-npu-dai/mova/diffusion/pipelines/mova_train.py, line 1105, "
            "in _fn, \n return module(*inputs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/utils/checkpoint.py, line 1555, "
            "in recompute_fn, \n fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/utils/checkpoint.py, line 1124, "
            "in _run_fn_with_dynamo_disabled, \n return fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py, line 1044, "
            "in _fn, \n return fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/_compile.py, line 53, in inner, "
            "\n return disable_fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/utils/checkpoint.py, line 1154, "
            "in unpack_hook, \n _run_fn_with_dynamo_disabled(frame.recompute_fn, *args)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/utils/checkpoint.py, line 1147, "
            "in unpack_hook, \n args = ctx.get_args(ctx.saved_tensors)"
        ]
        self.assertTrue(GraphBuilder._is_recompute_by_stack_torch(stack_list))
        stack_list1 = [
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1840, "
            "in inner, \n hook_result = hook(self, args, kwargs, result)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1881, "
            "in _call_impl, \n return inner()",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1775, "
            "in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /root/work/filestorage/gh/code/MOVA-feat-npu-dai/mova/diffusion/models/wan_video_dit.py, line 242, "
            "in forward, \n v = self.v(ctx)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1829, "
            "in inner, \n result = forward_call(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1881, "
            "in _call_impl, \n return inner()",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1775, "
            "in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/utils/checkpoint.py, line 1555, "
            "in recompute_fn, \n fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/utils/checkpoint.py, line 1124, "
            "in _run_fn_with_dynamo_disabled, \n return fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py, line 1044, "
            "in _fn, \n return fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/_compile.py, line 53, in inner, "
            "\n return disable_fn(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/utils/checkpoint.py, line 1154, "
            "in unpack_hook, \n _run_fn_with_dynamo_disabled(frame.recompute_fn, *args)"
        ]
        self.assertTrue(GraphBuilder._is_recompute_by_stack_torch(stack_list1))
        stack_list2 = [
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1840, "
            "in inner, \n hook_result = hook(self, args, kwargs, result)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1881, "
            "in _call_impl, \n return inner()",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1775, "
            "in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /root/.local/lib/python3.11/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py, "
            "line 599, in forward, \n x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1829, "
            "in inner, \n result = forward_call(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1881, "
            "in _call_impl, \n return inner()",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1775, "
            "in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /root/.local/lib/python3.11/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py, "
            "line 1142, in _encode, \n out_ = self.encoder(",
            "File /root/.local/lib/python3.11/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py, "
            "line 1173, in encode, \n h = self._encode(x)",
            "File /root/.local/lib/python3.11/site-packages/diffusers/utils/accelerate_utils.py, line 46, in wrapper, "
            "\n return method(self, *args, **kwargs)",
            "File /root/work/filestorage/gh/code/MOVA-feat-npu-dai/mova/diffusion/pipelines/mova_train.py, line 1344, "
            "in training_step, \n video_latents = self.video_vae.encode(video).latent_dist.mode()",
            "File /root/work/filestorage/gh/code/MOVA-feat-npu-dai/mova/diffusion/pipelines/mova_train.py, line 1279, "
            "in forward, \n return self.training_step(*args, cp_mesh=cp_mesh, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1829, "
            "in inner, \n result = forward_call(*args, **kwargs)",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1881, "
            "in _call_impl, \n return inner()",
            "File /root/.local/conda/envs/mova/lib/python3.11/site-packages/torch/nn/modules/module.py, line 1775, "
            "in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
            "File /root/work/filestorage/gh/code/MOVA-feat-npu-dai/mova/engine/trainer/accelerate/accelerate_trainer.py"
            ", line 414, in train, \n loss_dict = self.model(",
            "File /root/work/filestorage/gh/code/MOVA-feat-npu-dai/scripts/training_scripts/accelerate_train.py, "
            "line 180, in main, \n trainer.train()",
            "File /root/work/filestorage/gh/code/MOVA-feat-npu-dai/scripts/training_scripts/accelerate_train.py, "
            "line 184, in <module>, \n main()"
        ]
        self.assertFalse(GraphBuilder._is_recompute_by_stack_torch(stack_list2))
