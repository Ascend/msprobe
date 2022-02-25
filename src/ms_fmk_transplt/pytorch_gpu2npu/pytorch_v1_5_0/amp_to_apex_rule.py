from typing import Optional, Union

import libcst
from libcst import matchers as m, FlattenSentinel, RemovalSentinel

from pytorch_gpu2npu.common_rules import InsertGlobalRule
from pytorch_gpu2npu.common_rules.code_visitor import RuleVisitor, OperatorType
from pytorch_gpu2npu.utils.scope_visitors import ScaleScopeVisitor


class InitApexRule(InsertGlobalRule):
    def __init__(self):
        insert_content = ["from apex import amp"]
        super(InitApexRule, self).__init__(insert_content)
        self.insert_flag = False

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        qualified_name = self.get_full_name_for_node(node)
        if qualified_name == 'torch.cuda.amp.autocast':
            self.insert_flag = True
        return True

    def clean(self):
        super().clean()
        self.insert_flag = False


class Amp2Apex(RuleVisitor):
    """
    Convert torch.cuda.amp to apex.amp
    """

    def __init__(self, model, main_name):
        super(Amp2Apex, self).__init__()
        self.scaler_name = ''
        self.loss_name = ''
        self.optimizer_name = ''
        self.model_name = model
        self.main_file_name = main_name
        self.delete_scaler_update = False
        self.delete_scaler_loss = False
        self.delete_scaler_optimizer = False
        self.delete_scaler_gardscaler = False
        self.find_optimizer = False
        self.find_model = False
        self.model_ddp = None

    def visit_Module(self, node: "libcst.Module") -> Optional[bool]:
        visitor = ScaleScopeVisitor()
        wrapper = libcst.metadata.MetadataWrapper(node)
        wrapper.visit(visitor)
        self.loss_name = visitor.loss_name
        self.optimizer_name = visitor.optimizer_name
        self.scaler_name = visitor.scaler_name

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        super().visit_Assign(node)
        if not m.matches(node.value, m.Call()):
            return True
        if self.get_full_name_for_node(node.value) == "torch.cuda.amp.GradScaler":
            self.delete_scaler_gardscaler = True
            self._record_position(node, OperatorType.DELETE,
                                  "delete the torch.cuda.amp.Gradscaler statement")
        return True

    def __adapt_model_ddp(self, original_node, updated_nodes):
        """
        adjust the position between ddp(model) and optimizer declaration
        """
        if not self.scaler_name:
            return
        model_ddp_list = ('torch.nn.parallel.DistributedDataParallel', 'torch.nn.DataParallel')
        if not m.matches(original_node.body[0], m.Assign(value=m.Call())) or \
                self.get_full_name_for_node(original_node.body[0].value) not in model_ddp_list:
            return
        self.find_model = True
        if self.find_optimizer:
            self.find_optimizer = False
            return
        self.model_ddp = original_node
        updated_nodes.pop()

    def __generator_apex_initialize(self, original_node, updated_nodes):
        """
        Generate apex.amp initialization code
        """
        if not m.matches(original_node.body[0], m.Assign(value=m.Call())) or len(self.optimizer_name) == 0:
            return
        target = original_node.body[0].targets[0].target
        if self.get_full_name_for_node(target, with_variable_replace=False) != self.optimizer_name:
            return
        apex_initialize_statement = libcst.parse_statement(
            '%s, %s = amp.initialize(%s, %s, opt_level="O1", loss_scale="32")'
            % (self.model_name, self.optimizer_name, self.model_name, self.optimizer_name))
        updated_nodes.append(apex_initialize_statement)
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line + 1,
                                  original_position.start.line + 1,
                                  OperatorType.INSERT.name, "init statement of apex"])
        if self.main_file_name:
            ddp_statement = libcst.parse_statement(
                'if not isinstance(%s, torch.nn.parallel.DistributedDataParallel):\n'
                '    %s = torch.nn.parallel.DistributedDataParallel(%s, device_ids=[NPU_CALCULATE_DEVICE], '
                'broadcast_buffers=False)' % (self.model_name, self.model_name, self.model_name))
            updated_nodes.append(ddp_statement)
            self.changes_info.append([original_position.start.line + 1,
                                      original_position.start.line + 3,
                                      OperatorType.INSERT.name, "init statement of DistributedDataParallel"])
        if self.find_model:
            self.find_optimizer = True
            self.find_model = False
            updated_nodes.append(self.model_ddp)

    def __remove_torch_cuda_amp(self, original_node, updated_nodes):
        """
        Delete the import of torch.cuda.amp
        """
        if not m.matches(original_node.body[0], m.Import()) and not m.matches(original_node.body[0], m.ImportFrom()):
            return
        if m.matches(original_node.body[0].names, m.ImportStar()):
            return
        if self.get_full_name_for_node(original_node.body[0].names[0].name) == 'torch.cuda.amp':
            updated_nodes.pop()
        if self.get_full_name_for_node(original_node.body[0].children[1]) == "torch.cuda" and \
                self.get_full_name_for_node(original_node.body[0].names[0].name) == 'amp':
            updated_nodes.pop()

    def __delete_scaler_loss(self, updated_nodes):
        """
        Delete scale.loss()
        """
        if self.delete_scaler_loss:
            self.delete_scaler_loss = False
            updated_nodes.pop()

    def __delete_scaler_optimizer(self, updated_nodes):
        """
        Delete scale.optimizer() and add with amp.scale_loss() code
        """
        if self.delete_scaler_optimizer:
            updated_nodes.pop()
            self.delete_scaler_optimizer = False
            if len(self.loss_name) == 0 or len(self.optimizer_name) == 0:
                return
            apex_loss_statement = libcst.parse_statement(
                'with amp.scale_loss(%s, %s) as scaled_loss:\n'
                '   scaled_loss.backward()\n' % (self.loss_name, self.optimizer_name)
            )
            optimizer_statement = libcst.parse_statement('%s.step()' % (self.optimizer_name))
            updated_nodes.append(apex_loss_statement)
            updated_nodes.append(optimizer_statement)

    def __delete_scaler_gardscaler(self, updated_nodes):
        """
        Delete torch.cuda.amp.GradScaler()
        """
        if self.delete_scaler_gardscaler:
            self.delete_scaler_gardscaler = False
            updated_nodes.pop()

    def __delete_scaler_update(self, updated_nodes):
        """
        Delete scaler.update()
        """
        if self.delete_scaler_update:
            self.delete_scaler_update = False
            updated_nodes.pop()

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        qualified_name = self.get_full_name_for_node(node)
        scale_str = f'{self.scaler_name}.scale'
        optimizer_str = f'{self.scaler_name}.step'
        update_str = f'{self.scaler_name}.update'
        if qualified_name == scale_str:
            self.delete_scaler_loss = True
            self._record_position(node, OperatorType.DELETE,
                                  "delete the scaler scale statement")
        if qualified_name == optimizer_str:
            self.delete_scaler_optimizer = True
            self._record_position(node, OperatorType.MODIFY,
                                  "change the scaler.step() to optimizer.step()")
        if qualified_name == update_str:
            self.delete_scaler_update = True
            self._record_position(node, OperatorType.DELETE,
                                  "delete the scaler update statement")
        return True

    def leave_SimpleStatementLine(self, original_node: "libcst.SimpleStatementLine",
                                  updated_node: "libcst.SimpleStatementLine"
                                  ) -> Union["libcst.BaseStatement",
                                             FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:

        updated_nodes = [updated_node]
        self.__generator_apex_initialize(original_node, updated_nodes)
        self.__adapt_model_ddp(original_node, updated_nodes)
        self.__delete_scaler_loss(updated_nodes)
        self.__delete_scaler_optimizer(updated_nodes)
        self.__delete_scaler_gardscaler(updated_nodes)
        self.__delete_scaler_update(updated_nodes)
        self.__remove_torch_cuda_amp(original_node, updated_nodes)

        if len(updated_nodes) == 1:
            return updated_node
        elif len(updated_nodes) != 0:
            return libcst.FlattenSentinel(updated_nodes)
        else:
            return libcst.RemovalSentinel.REMOVE

    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        if not self.scaler_name:
            return updated_node
        qualified_name = self.get_full_name_for_node(original_node)
        model_ddp_list = ('torch.nn.parallel.DistributedDataParallel', 'torch.nn.DataParallel')
        if qualified_name not in model_ddp_list:
            return updated_node
        return updated_node.with_changes(args=[])

    def leave_With(self, original_node: "libcst.With", updated_node: "libcst.With") \
            -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        item = original_node.items[0].item
        if not m.matches(item, m.Call()):
            return updated_node
        if self.get_full_name_for_node(item) == "torch.cuda.amp.autocast":
            loss_statement = updated_node.body.body
            return libcst.FlattenSentinel(loss_statement)
        return updated_node

    def clean(self):
        super().clean()
        self.scaler_name = ''
        self.loss_name = ''
        self.optimizer_name = ''
        self.delete_scaler_update = False
        self.delete_scaler_loss = False
        self.delete_scaler_optimizer = False
        self.delete_scaler_gardscaler = False
        self.find_optimizer = False
        self.find_model = False
        self.model_ddp = None