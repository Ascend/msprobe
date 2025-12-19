#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import unittest
import libcst
import sys
import os

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))


class TestRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import src.ms_fmk_transplt.transfer.rules.common_rules.common_rule as common_rule
        import src.ms_fmk_transplt.transfer.rules.distributed_rules.distributed_rule as distributed_rule
        import src.ms_fmk_transplt.transfer.rules.modelarts_rules as modelarts_rule
        from src.ms_fmk_transplt.transfer.rules.pytorch_npu_patch_rules import insert_ahead_rule as rule_1_8_1
        from src.ms_fmk_transplt.utils import trans_utils as utils
        cls.common_rule = common_rule
        cls.distributed_rule = distributed_rule
        cls.modelarts_rule = modelarts_rule
        cls.utils = utils
        cls.rule_1_8_1 = rule_1_8_1

    def test_args_modify_rule(self):
        load_rule = self.common_rule.ArgsModifyRule('torch.load', '"npu:0"', -1, 'map_location', ['cpu'])
        normal_rule = self.common_rule.ArgsModifyRule('func', '"npu:0"', 0)
        arg_delete_rule = self.common_rule.ArgsModifyRule('func', '', 1)
        arg_keyword_delete_rule = self.common_rule.ArgsModifyRule('torch.profiler.profile', '', -1,
                                                                  'experimental_config')
        to_rule = self.common_rule.ArgsModifyRule('to', "f'npu:{replace_device_int}' if "
                                                        "isinstance(replace_device_int, int) else replace_device_int",
                                                  0)

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

        arg_keyword_delete_case = (
            (
                "torch.profiler.profile(with_flops=False, with_modules=False, experimental_config = config)",
                "torch.profiler.profile(with_flops=False, with_modules=False, )"
            ),
            (
                "torch.profiler.profile(with_flops=False)",
                "torch.profiler.profile(with_flops=False)"
            )
        )
        for test_case in arg_keyword_delete_case:
            self._check_modify(arg_keyword_delete_rule, test_case[0], test_case[1])

            # 2.1 device type int replace
        device_int_cases = (
            ("a.to(d)", "a.to(f'npu:{d}' if isinstance(d, int) else d)"),
            (
                "new_order.to(src_tokens.device)",
                "new_order.to(f'npu:{src_tokens.device}' if isinstance(src_tokens.device, int) else src_tokens.device)"
            )
        )
        for test_case in device_int_cases:
            self._check_modify(to_rule, test_case[0], test_case[1])

    def test_insert_global_rule(self):
        rule = self.common_rule.InsertGlobalRule(["import key", "key.insert()"])
        test_cases = (("import torch\ntest_case_with_key_word()",
                       "import torch\nimport key\nkey.insert()\ntest_case_with_key_word()"),
                      ("from torch import nn\ntest_case_with_key_word()",
                       "from torch import nn\nimport key\nkey.insert()\ntest_case_with_key_word()"),
                      ("import numpy\ntest_case_without_key_word()",
                       "import numpy\nimport key\nkey.insert()\ntest_case_without_key_word()")
                      )
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])
            rule.clean()

    def test_insert_ahead_rule(self):
        rule = self.rule_1_8_1.InsertAheadRule()
        test_cases = (
            (
                '''import torch.npu
from torch.npu import amp
                ''', '''import torch_npu
import torch.npu
from torch.npu import amp
                '''
            ), (
                '''import os
from torch.npu import amp
                ''', '''import os
import torch_npu
from torch.npu import amp
                '''
            ), (
                '''import os
import sys
                ''', '''import os
import sys
                '''
            )
        )
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])
            rule.clean()

    def test_insert_main_file_rule(self):
        init_process_group_content = ["import torch.npu",
                                      "if torch.npu.current_device() != DEVICE_ID:\n"
                                      "    torch.npu.set_device(f'npu:{DEVICE_ID}')",
                                      "RANK_SIZE = int(os.getenv('RANK_SIZE'))",
                                      "RANK_ID = int(os.getenv('RANK_ID'))",
                                      "torch.distributed.init_process_group('hccl', rank=RANK_ID, world_size=RANK_SIZE)"]
        rule = self.common_rule.InsertMainFileRule(init_process_group_content)
        test_cases = (
            (
                '''import torch

def train():
    pass

if __name__ == '__main__':
    train()
                ''',
                '''import torch
import torch.npu
if torch.npu.current_device() != DEVICE_ID:
    torch.npu.set_device(f'npu:{DEVICE_ID}')
RANK_SIZE = int(os.getenv('RANK_SIZE'))
RANK_ID = int(os.getenv('RANK_ID'))
torch.distributed.init_process_group('hccl', rank=RANK_ID, world_size=RANK_SIZE)

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
            rule.clean()

    def test_func_name_modify_rule(self):
        rule = self.common_rule.FuncNameModifyRule("old_name", "new_name", False)
        test_cases = (("old_name()", "new_name()"),
                      ("AA.old_name()", "AA.new_name()"),
                      ("AA.BB.old_name()", "AA.BB.new_name()"),
                      ("AA.old_name.BB(old_name())", "AA.old_name.BB(new_name())"),
                      ("AA.old_name.old_name()", "AA.old_name.new_name()"),
                      ("(other_name if xxx else old_name)()", "(other_name if xxx else new_name)()"),
                      ("(other_name if xxx else (other_name if xxx else old_name))()",
                       "(other_name if xxx else (other_name if xxx else new_name))()"),
                      ("(pids1 == pids2).long().old_name()", "(pids1 == pids2).long().new_name()"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

        rule = self.common_rule.FuncNameModifyRule("old_name", "AA.BB.new_name", True)
        test_cases = (("old_name()", "AA.BB.new_name()"),
                      ("AA.old_name()", "AA.BB.new_name()"),
                      ("AA.old_name.BB(old_name())", "AA.old_name.BB(AA.BB.new_name())"),
                      ("DD.old_name.old_name()", "AA.BB.new_name()"),
                      ("(other_name if xxx else old_name)()", "(other_name if xxx else AA.BB.new_name)()"),
                      ("(other_name if xxx else (other_name if xxx else old_name))()",
                       "(other_name if xxx else (other_name if xxx else AA.BB.new_name))()"))
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

        rule = self.common_rule.FuncNameModifyRule("CC.DD.old_name", "AA.BB.new_name", True)
        test_cases = (("CC.DD.old_name()", "AA.BB.new_name()"),
                      ("import CC.DD as CD\nCD.old_name()", "import CC.DD as CD\nAA.BB.new_name()"),
                      ("CD = CC.DD\nCD.old_name()", "CD = CC.DD\nAA.BB.new_name()"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_module_name_modify_rule(self):
        rule = self.common_rule.ModuleNameModifyRule("old_name", "new_name", "AA.BB")

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
        strict_rule = self.common_rule.ReplaceStringRule("old_str", "new_str", True)
        normal_rule = self.common_rule.ReplaceStringRule("old_str", "new_str", False)

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
        rule = self.common_rule.ReplaceAttributeRule("old_name", "new_name")

        test_cases = (("a = func()\na.old_name", "a = func()\na.new_name"),
                      ("func().old_name", "func().new_name"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

        rule = self.common_rule.ReplaceAttributeRule("torch.profiler.ProfilerAction",
                                                     "torch_npu.profiler.ProfilerAction")

        test_case = ("torch.profiler.ProfilerAction", "torch_npu.profiler.ProfilerAction")

        self._check_modify(rule, test_case[0], test_case[1])

    def test_python_version_convert_rule(self):
        rule = self.common_rule.PythonVersionConvertRule()

        test_cases = (("hasattr(model.module, 'optimizer')", "hasattr(model.modules, 'optimizer')"),
                      ("if hasattr(model.module, 'optimizer'):\n    pass",
                       "if hasattr(model.modules, 'optimizer'):\n    pass"))

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_dataloader_rule(self):
        rule = self.distributed_rule.DataLoaderRule()

        test_cases = (
            (
                '''from torch.utils import data

trainset = ICDAR15(args.train_data,args.train_gt)
train_loader_target = data.DataLoader(trainset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, 
                                      drop_last=True)
f_score = 0.5
for epoch in range(args.epoch_iter):
    train(epoch, model, optimizer,train_loader_target,criterion)
            ''',
            '''from torch.utils import data

trainset = ICDAR15(args.train_data,args.train_gt)
train_loader_target = data.DataLoader(trainset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers, 
                                      drop_last=True, pin_memory = True, sampler = torch.utils.data.distributed.DistributedSampler(trainset))
f_score = 0.5
for epoch in range(args.epoch_iter):
    train_loader_target.sampler.set_epoch(epoch)
    train(epoch, model, optimizer,train_loader_target,criterion)
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
            ), (
                '''from torch.utils.data.dataloader import DataLoader
dataloader = None
dataloader = DataLoader(train_data, sampler=train_sampler, 
                        batch_size=args.train_batch_size)
                ''', '''from torch.utils.data.dataloader import DataLoader
dataloader = None
dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle = False, pin_memory = True, drop_last = True, sampler = torch.utils.data.distributed.DistributedSampler(train_data))
                '''
            ), (
                '''import torch

dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)
        for x in ['train', 'valid']}

for epoch in epochs:
    pass
                ''', '''import torch

dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory = True, drop_last = True, sampler = torch.utils.data.distributed.DistributedSampler(face_dataset[x]))
        for x in ['train', 'valid']}

for epoch in epochs:
    for loader in dataloaders.values():
        loader.sampler.set_epoch(epoch)
    pass
                '''
            ), (
                '''import torch

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
                        ''', '''import torch

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last = True, sampler = torch.utils.data.distributed.DistributedSampler(datasets.ImageFolder(valdir, val_transforms)))
                        '''
            )
        )

        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_distributed_data_parallel_rule(self):
        rule = self.distributed_rule.DistributedDataParallelRule('model')

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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ID], broadcast_buffers=False)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)
                '''
            ), (
                '''model, dataset = EAST(pretrained=False), DateSet()

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)
                ''',
                '''model, dataset = EAST(pretrained=False), DateSet()
model = model.npu()
if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ID], broadcast_buffers=False)

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
model = model.cuda()
model = model.npu()
model = model.to(device)
model, opt = amp.initialize(model, opt)
model = model.npu()
if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ID], broadcast_buffers=False)
                '''
            ), (
                '''import torch
if config['deep_supervision']:
    output = model(input)[-1]
else:
    output = model(input)
                ''', '''import torch
if config['deep_supervision']:
    output = model(input)[-1]
else:
    output = model(input)
                '''
            ), (
                '''import torch
from apex import amp

model = EAST()
if args.fp16:
    model, opt = amp.initialize(model, opt)
if args.resume:
    model = model.load_state_dict('model.npz')
    ''', '''import torch
from apex import amp

model = EAST()
if args.fp16:
    model, opt = amp.initialize(model, opt)
model = model.npu()
if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ID], broadcast_buffers=False)
if args.resume:
    model = model.module.load_state_dict('model.npz')
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
        rule = self.common_rule.FuncNameModifyRule("functionA", "FUNCTIONA", False)
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def test_modelarts_path_warpper_rule(self):
        path_handler_api_dict = {
            'builtins.open': {
                'arg_no': 0,
            },
            'torch.save': {
                'arg_no': 1,
            },
            'shutil.copyfile': {
                'arg_no': [0, 1],
            },
        }
        add_import_rule = self.common_rule.BaseInsertGlobalRule(insert_content=[])
        path_wrapper_rule = self.modelarts_rule.ModelArtsPathWrapperRule(path_handler_api_dict, add_import_rule)
        test_cases = (
            (
                '''import shutil
import torch

def save_checkpoint(state, is_best, filename='checkpoint.ckpt'):
    torch.save(state, filename)
    data = [(str.title if meta["species"] == "cat" else str.lower)(part) for part in meta["cls"].split()]
    if is_best:
        shutil.copyfile(filename, dst='model_best.ckpt')

torch.save(data, open('traindata.pt', 'wb'))
                ''', '''import shutil
import torch

def save_checkpoint(state, is_best, filename='checkpoint.ckpt'):
    torch.save(state, ModelArtsPathManager().get_path(filename))
    data = [(str.title if meta["species"] == "cat" else str.lower)(part) for part in meta["cls"].split()]
    if is_best:
        shutil.copyfile(ModelArtsPathManager().get_path(filename), dst=ModelArtsPathManager().get_path('model_best.ckpt'))

torch.save(data, ModelArtsPathManager().get_path(open(ModelArtsPathManager().get_path('traindata.pt'), 'wb')))
                '''
            ),
        )

        for test_case in test_cases:
            self._check_modify(path_wrapper_rule, test_case[0], test_case[1])

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
        rule = self.common_rule.FuncNameModifyRule("cuda", "npu", False)
        self._check_modify(rule, test_cases1[0][0], test_cases1[0][1])
        rule = self.common_rule.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[1][0], test_cases1[1][1])
        rule = self.common_rule.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[2][0], test_cases1[2][1])
        rule = self.common_rule.FuncNameModifyRule("cuda", "npu", False)
        self._check_modify(rule, test_cases1[3][0], test_cases1[3][1])
        rule = self.common_rule.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[4][0], test_cases1[4][1])
        rule = self.common_rule.FuncNameModifyRule("cuda", "torch1.npu", True)
        self._check_modify(rule, test_cases1[5][0], test_cases1[5][1])
        rule = self.common_rule.FuncNameModifyRule("cuda", "torch1.npu", True)
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
        rule = self.common_rule.FuncNameModifyRule("cuda", "npu", False)
        for test_case in test_cases:
            self._check_modify(rule, test_case[0], test_case[1])

    def _check_modify(self, rule, code, expected_result):
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        new_module = wrapper.visit(rule)
        self.assertEqual(expected_result, new_module.code)

    def test_tran_utils(self):
        file = os.path.realpath(__file__)

        # remove_path
        dir_path = './test_delete'
        os.makedirs(dir_path)
        os.chmod(dir_path, 0o000)
        self.assertRaises(self.utils.DeleteFileException, self.utils.remove_path, dir_path)
        os.chmod(dir_path, 0o700)
        self.utils.remove_path(dir_path)

        # get_main_file
        main_file = self.utils.get_main_file(file, file)
        self.assertEqual(main_file, os.path.basename(file))

        # name_to_jedi_position
        position = self.utils.name_to_jedi_position(file, 22, 'os')
        self.assertNotEquals(position, {})
        position = self.utils.name_to_jedi_position(file, 10000, 'os')
        self.assertEqual(position, {})
        position = self.utils.name_to_jedi_position(file, 12, 'os')
        self.assertEqual(position, {})

        # walk_input_path
        project = os.path.join(os.path.dirname(__file__), '../resources/net/barlowtwins_amp')
        py_file_counts = self.utils.walk_input_path(project, output_free_size=1024 ** 3)
        self.assertEqual(py_file_counts, 3)

        # check_model_name_valid
        invalid_model_names = ['123', '12model', '{}', 'model*#']
        for invalid_model_name in invalid_model_names:
            self.assertRaises(ValueError, self.utils.check_model_name_valid, invalid_model_name)

        valid_model_names = ['model', '_model', 'self.model', 'self.model1_']
        for valid_model_name in valid_model_names:
            self.assertEqual(self.utils.check_model_name_valid(valid_model_name), None)


if __name__ == '__main__':
    unittest.main()
