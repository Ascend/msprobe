#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import unittest
import coverage
import libcst
import sys
import os

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))


class TestRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import src.ms_fmk_transplt.rule as rule_module
        cls.rule_module = rule_module


    def test_args_modify_rule(self):
        load_rule = self.rule_module.ArgsModifyRule('torch.load', '"npu:0"', -1, 'map_location', ['cpu'])
        normal_rule = self.rule_module.ArgsModifyRule('func', '"npu:0"', 0)
        arg_delete_rule = self.rule_module.ArgsModifyRule('func', '', 1)

        load_cases = (
            # map_location not specified
            ("torch.load('pretrained.pt')",
             "torch.load('pretrained.pt')"),
            ("torch.load('pretrained.pt', pickle_module=dummy_pickle)",
             "torch.load('pretrained.pt', pickle_module=dummy_pickle)"),
            ("torch.load('pretrained.pt', pickle_module=dummy_pickle, var_arg=foo)",
             "torch.load('pretrained.pt', pickle_module=dummy_pickle, var_arg=foo)"),
            # with keyword
            ("torch.load('pretrained.pt', map_location='cuda:0')",
             "torch.load('pretrained.pt', map_location=\"npu:0\")"),
            ("torch.load('pretrained.pt', pickle_module=dummy_pickle, map_location='cuda:0')",
             "torch.load('pretrained.pt', pickle_module=dummy_pickle, map_location=\"npu:0\")"),
            # whitelist
            ("torch.load('pretrained.pt', map_location='cpu')",
             "torch.load('pretrained.pt', map_location='cpu')")
        )

        for test_case in load_cases:
            self._check_modify(load_rule, test_case[0], test_case[1])

        normal_case = (("func('cuda', args)", "func(\"npu:0\", args)"),
                       ("funcA('cuda', args)", "funcA('cuda', args)"))
        for test_case in normal_case:
            self._check_modify(normal_rule, test_case[0], test_case[1])

        arg_delete_cases = (("func('npu', args)", "func('npu', )"),
                            ("funcA('npu', args)", "funcA('npu', args)"))
        for test_case in arg_delete_cases:
            self._check_modify(arg_delete_rule, test_case[0], test_case[1])

    def test_specify_device_insert_rule(self):
        rule_with_key_word = self.rule_module.InsertGlobalRule(["import key", "key.insert()"], "torch")
        rule_without_key_word = self.rule_module.InsertGlobalRule(["import key", "key.insert()"], "")
        test_cases = (("import torch\ntest_case_with_key_word()",
                       "import torch\nimport key\nkey.insert()\ntest_case_with_key_word()",
                       "import torch\nimport key\nkey.insert()\ntest_case_with_key_word()"),
                      ("from torch import nn\ntest_case_with_key_word()",
                       "from torch import nn\nimport key\nkey.insert()\ntest_case_with_key_word()",
                       "from torch import nn\nimport key\nkey.insert()\ntest_case_with_key_word()"),
                      ("import numpy\ntest_case_without_key_word()",
                       "import numpy\ntest_case_without_key_word()",
                       "import numpy\nimport key\nkey.insert()\ntest_case_without_key_word()")
                      )
        for test_case in test_cases:
            self._check_modify(rule_with_key_word, test_case[0], test_case[1])
            self._check_modify(rule_without_key_word, test_case[0], test_case[2])
            rule_with_key_word.clean()
            rule_without_key_word.clean()

    def test_func_name_modify_rule(self):
        rule = self.rule_module.FuncNameModifyRule("old_name", "new_name", False)
        test_cases = (("old_name()", "new_name()"),
                      ("AA.old_name()", "AA.new_name()"),
                      ("AA.BB.old_name()", "AA.BB.new_name()"),
                      ("AA.old_name.BB(old_name())", "AA.old_name.BB(new_name())"),
                      ("AA.old_name.old_name()", "AA.old_name.new_name()"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

        rule = self.rule_module.FuncNameModifyRule("old_name", "AA.BB.new_name", True)
        test_cases = (("old_name()", "AA.BB.new_name()"),
                      ("AA.old_name()", "AA.BB.new_name()"),
                      ("AA.old_name.BB(old_name())", "AA.old_name.BB(AA.BB.new_name())"),
                      ("DD.old_name.old_name()", "AA.BB.new_name()"))
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

        rule = self.rule_module.FuncNameModifyRule("CC.DD.old_name", "AA.BB.new_name", True)
        test_cases = (("CC.DD.old_name()", "AA.BB.new_name()"),
                      ("import CC.DD as CD\nCD.old_name()", "import CC.DD as CD\nAA.BB.new_name()"),
                      ("CD = CC.DD\nCD.old_name()", "CD = CC.DD\nAA.BB.new_name()"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_module_name_modify_rule(self):
        rule = self.rule_module.ModuleNameModifyRule("old_name", "new_name", "AA.BB")

        test_cases = (("import AA.BB.old_name", "import AA.BB.new_name"),
                      ("import AA.BB.old_name as old\nold.func()", "import AA.BB.new_name as old\nold.func()"),
                      ("import AA.BB as AB\nAB.old_name", "import AA.BB as AB\nAB.new_name"),
                      ("AA.BB.old_name.func()", "AA.BB.new_name.func()"),
                      ("old_name.func()", "old_name.func()"),
                      ("CC.old_name.func()", "CC.old_name.func()"),
                      ("CC.DD.old_name()", "CC.DD.old_name()"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_replace_string_rule(self):
        strict_rule = self.rule_module.ReplaceStringRule("old_str", "new_str", True)
        normal_rule = self.rule_module.ReplaceStringRule("old_str", "new_str", False)

        test_cases = (("A = 'old_str'", "A = 'new_str'", "A = 'new_str'"),
                      ("A = \"old_str\"", "A = \"new_str\"", "A = \"new_str\""),
                      ("func(A = 'old_str')", "func(A = 'new_str')", "func(A = 'new_str')"),
                      ("# this is old_str", "# this is old_str", "# this is old_str"),
                      ("\"\"\"this is old_str\"\"\"", "\"\"\"this is old_str\"\"\"", "\"\"\"this is new_str\"\"\""),
                      ("func('old_str:%s' % tmp)", "func('old_str:%s' % tmp)", "func('new_str:%s' % tmp)"),
                      ("import old_str", "import old_str", "import old_str"),
                      ("A = f'old_str{abc}'", "A = f'old_str{abc}'", "A = f'new_str{abc}'"))

        for test_case in test_cases:
            self._check_modify(strict_rule, test_case[0], test_case[1])
            self._check_modify(normal_rule, test_case[0], test_case[2])

    def test_replace_attribute_rule(self):
        # this rule will replace import module and function name
        rule = self.rule_module.ReplaceAttributeRule("old_name", "new_name")

        test_cases = (("a = func()\na.old_name", "a = func()\na.new_name"),
                      ("func().old_name", "func().new_name"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_python_version_convert_rule(self):
        rule = self.rule_module.PythonVersionConvertRule()

        test_cases = (("hasattr(model.module, 'optimizer')", "hasattr(model.modules, 'optimizer')"),
                      ("if hasattr(model.module, 'optimizer'):\n    pass",
                       "if hasattr(model.modules, 'optimizer'):\n    pass"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_init_process_group_rule(self):
        rule = self.rule_module.InitProcessGroupRule()

        test_cases = (
            (
                '''import torch

def train():
    pass

if __name__ == '__main__':
    train()
                ''',
                '''import torch
NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))
RANK = int(os.getenv('RANK'))
torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)

def train():
    pass

if __name__ == '__main__':
    train()
                '''
            ),
        )

        for test_case in test_cases:
            rule.visit_main_file(True)
            self._check_modify(rule, test_case[0], test_case[1])


    def test_dataloader_rule(self):
        rule = self.rule_module.DataLoaderRule()

        test_cases = (
            (
                '''from torch.utils import data

trainset = ICDAR15(args.train_data,args.train_gt)
train_loader_target = data.DataLoader(trainset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, drop_last=True)
f_score = 0.5
for epoch in range(args.epoch_iter):
    train( epoch, model, optimizer,train_loader_target,criterion)
            ''',
            '''from torch.utils import data

trainset = ICDAR15(args.train_data,args.train_gt)
train_loader_target_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
train_loader_target_batch_size = max(int(args.batch_size / int(os.getenv('NPU_WORLD_SIZE'))), 1)
train_loader_target = data.DataLoader(trainset, batch_size=train_loader_target_batch_size,
                                      shuffle=False, num_workers=args.num_workers, drop_last=True, pin_memory = True, sampler = train_loader_target_sampler)
f_score = 0.5
for epoch in range(args.epoch_iter):
    train_loader_target.sampler.set_epoch(epoch)
    train( epoch, model, optimizer,train_loader_target,criterion)
            '''), (
                '''from torch.utils import data

def train(data_loader):
    for epoch in args.epoch:
        model.train()
            ''',
            '''from torch.utils import data

def train(data_loader):
    for epoch in args.epoch:
        if isinstance(data_loader, torch.utils.data.DataLoader):
            data_loader.sampler.set_epoch(epoch)
        model.train()
            '''
            )
        )

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_distributed_data_parallel_rule(self):
        rule = self.rule_module.DistributedDataParallelRule('model', '')

        test_cases = (
            (
                '''model = EAST(pretrained=False)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)
                ''',
                '''model = EAST(pretrained=False)
model = model.npu()
if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)
                '''
            ),(
                '''model, dataset = EAST(pretrained=False), DateSet()

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)
                ''',
                '''model, dataset = EAST(pretrained=False), DateSet()
model = model.npu()
if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)
                '''
            ), (
                '''import torch
from apex import amp

model = EAST()
model = model.cuda()
model = model.npu()
model = model.to(device)
model, opt = amp.initialize(model, opt)
model = torch.nn.DataParallel(model)
                ''',
                '''import torch
from apex import amp

model = EAST()
model = model.npu()
if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)
model = model.cuda()
model = model.npu()
model = model.to(device)
model, opt = amp.initialize(model, opt)
                '''
            )
        )

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_If_Exp_rule(self):
        test_cases = (('''def functionA(args):
    print("functionA ", args)

def functionB(args):
    print("functionB ", args)

def functionC(args):
    print("functionC ", args)

(functionA if True else functionA)("666")''', '''def functionA(args):
    print("functionA ", args)

def functionB(args):
    print("functionB ", args)

def functionC(args):
    print("functionC ", args)

(FUNCTIONA if True else FUNCTIONA)("666")'''),
                      ('''def functionA(args):
    print("functionA ", args)

def functionB(args):
    print("functionB ", args)

def functionC(args):
    print("functionC ", args)

(functionA if True else functionA if True else functionA)("666")''', '''def functionA(args):
    print("functionA ", args)

def functionB(args):
    print("functionB ", args)

def functionC(args):
    print("functionC ", args)

(FUNCTIONA if True else FUNCTIONA if True else FUNCTIONA)("666")'''),
                      ('''def functionA(args):
    print("functionA ", args)

def functionB(args):
    print("functionB ", args)

def functionC(args):
    print("functionC ", args)

(functionA if True else functionB if True else functionA)("666")''', '''def functionA(args):
    print("functionA ", args)

def functionB(args):
    print("functionB ", args)

def functionC(args):
    print("functionC ", args)

(FUNCTIONA if True else functionB if True else FUNCTIONA)("666")'''))
        rule = self.rule_module.FuncNameModifyRule("functionA", "FUNCTIONA", False)
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_If_Exp_rule1(self):
        test_cases1 = (('''(torch.cuda if True else torch.cuda if True else torch.cuda)(666)''',
                        '''(torch.npu if True else torch.npu if True else torch.npu)(666)'''),
                       ('''(torch.cuda if True else torch.cuda if True else torch.cuda)(666)''',
                        '''(torch1.npu if True else torch1.npu if True else torch1.npu)(666)'''),
                       ('''(cuda if True else cuda)(666)''',
                        '''(torch1.npu if True else torch1.npu)(666)'''),
                       ('''(torch.m.n.cuda if True else torch.m.n.cuda if True else torch.m.n.cuda)(666)''',
                        '''(torch.m.n.npu if True else torch.m.n.npu if True else torch.m.n.npu)(666)'''),
                       ('''(torch.m.n.cuda if True else torch.m.n.cuda if True else torch.m.n.cuda)(666)''',
                        '''(torch1.npu if True else torch1.npu if True else torch1.npu)(666)'''),
                       ('''(torch.m.n.cuda if True else cuda1 if True else torch.m.n.cuda)(666)''',
                        '''(torch1.npu if True else cuda1 if True else torch1.npu)(666)'''),
                       ('''(torch.m.n.cuda if torch.m.n.cuda() else cuda1 if True else torch.m.n.cuda)(666)''',
                        '''(torch1.npu if torch1.npu() else cuda1 if True else torch1.npu)(666)'''),
                       )
        rule = self.rule_module.FuncNameModifyRule("cuda", "npu", False)
        self._check_modify(rule, test_cases1[0][0], test_cases1[0][1])
        rule = self.rule_module.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[1][0], test_cases1[1][1])
        rule = self.rule_module.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[2][0], test_cases1[2][1])
        rule = self.rule_module.FuncNameModifyRule("cuda", "npu", False)
        self._check_modify(rule, test_cases1[3][0], test_cases1[3][1])
        rule = self.rule_module.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[4][0], test_cases1[4][1])
        rule = self.rule_module.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[5][0], test_cases1[5][1])
        rule = self.rule_module.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[6][0], test_cases1[6][1])

    def test_assign_definition(self):
        test_cases = (
            ('''student = student.cuda()
teacher = teacher.cuda()
pre1 = student(image)
pre2 = teacher(image)
''', '''student = student.npu()
teacher = teacher.npu()
pre1 = student(image)
pre2 = teacher(image)
'''), ('''student, teacher = student.cuda(), teacher.cuda()
pre1 = student(image)
pre2 = teacher(image)
''', '''student, teacher = student.npu(), teacher.npu()
pre1 = student(image)
pre2 = teacher(image)
''')
        )
        rule = self.rule_module.FuncNameModifyRule("cuda", "npu", False)
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_ascend_function(self):
        import torch
        import torch.nn.functional as F
        import src.ms_fmk_transplt.ascend_function.similar_api as sim_api
        in_tensor = torch.randn((4, 4, 5, 5, 5))
        torch_conv3d = torch.nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(2, 2, 2), dilation=2)
        conv_3d = sim_api.Conv3d(in_channels=4, out_channels=4, kernel_size=(2, 2, 2), dilation=2)
        torch_result = torch_conv3d(in_tensor)
        result = conv_3d(in_tensor)
        self.assertTrue(torch_result.shape == result.shape)

        self.assertTrue(type(sim_api.get_device_properties(device=1)) is sim_api.StubDeviceProperties)

        in_tensor = torch.Tensor([[[1, 2]]])
        out_tensor = torch.repeat_interleave(in_tensor, 2)
        result = sim_api.repeat_interleave(in_tensor, 2)
        self.assertTrue(out_tensor.equal(result))

        in_tensor = torch.Tensor([[[1, 2]]])
        out_tensor = torch.repeat_interleave(in_tensor, 2, dim=1)
        result = sim_api.repeat_interleave(in_tensor, 2, dim=1)
        self.assertTrue(out_tensor.equal(result))

        out_tensor = torch.repeat_interleave(in_tensor, torch.Tensor([2, 3]).long())
        result = sim_api.repeat_interleave(in_tensor, torch.Tensor([2, 3]).long())
        self.assertTrue(out_tensor.equal(result))

        in_tensor = torch.Tensor([[[1, 2]], [[3, 4]]])
        out_tensor = torch.repeat_interleave(in_tensor, torch.Tensor([2, 3]).long(), dim=0)
        result = sim_api.repeat_interleave(in_tensor, torch.Tensor([2, 3]).long(), dim=0)
        self.assertTrue(out_tensor.equal(result))

        in_tensor = torch.Tensor([[[1, 2]], [[3, 4]]])
        out_tensor = torch.repeat_interleave(in_tensor, torch.Tensor([2, 3]).long(), dim=2)
        result = sim_api.repeat_interleave(in_tensor, torch.Tensor([2, 3]).long(), dim=2)
        self.assertTrue(out_tensor.equal(result))

        in_tensor = torch.randn(4, 4, 4, 4)
        sync_batch_norm = sim_api.SyncBatchNorm(4)
        torch_sync_batch_norm = torch.nn.BatchNorm2d(4)
        result = sync_batch_norm(in_tensor)
        torch_result = torch_sync_batch_norm(in_tensor)
        self.assertTrue(torch_result.equal(result))

        in_tensor = torch.Tensor([[1, 2]])
        torch_result = F.pad(in_tensor, (1, 1))
        result = sim_api.pad(in_tensor, (1, 1))
        self.assertTrue(torch_result.equal(result))

        in_tensor = torch.Tensor([[[[1,2]]]])
        torch_result = F.pad(in_tensor, (1, 1))
        result = sim_api.pad(in_tensor, (1, 1))
        self.assertTrue(torch_result.equal(result))

        sim_api.set_default_tensor_type(torch.DoubleTensor)
        fp_64_tensor_1 = torch.zeros([2, 2])
        self.assertEqual(fp_64_tensor_1.dtype, torch.float64)

        sim_api.set_default_tensor_type('torch.npu.FloatTensor')
        fp_32_tensor_2 = torch.zeros([2, 2])
        self.assertEqual(fp_32_tensor_2.dtype, torch.float32)

        sim_api.set_default_tensor_type('torch.DoubleTensor')
        fp_64_tensor_2 = torch.zeros([2, 2])
        self.assertEqual(fp_64_tensor_2.dtype, torch.float64)

    def _check_modify(self, rule, code, expected_result):
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        new_module = wrapper.visit(rule)
        self.assertEqual(expected_result, new_module.code)


if __name__ == '__main__':
    unittest.main()
