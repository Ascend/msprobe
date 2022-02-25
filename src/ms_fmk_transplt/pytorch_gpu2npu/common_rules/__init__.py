from pytorch_gpu2npu.common_rules.code_visitor import ApiVisitor, OperatorType, RuleVisitor
from pytorch_gpu2npu.common_rules.common_rule import ArgsModifyRule, FuncNameModifyRule, InsertGlobalRule, \
    InsertMainFileRule, ModuleNameModifyRule, PythonVersionConvertRule, ReplaceAttributeRule, ReplaceStringRule
