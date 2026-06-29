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

from msprobe.core.common.const import Const
from msprobe.core.dump.data_dump.data_processor.base import BaseDataProcessor
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


class DataProcessorFactory:
    _data_processor = {}
    _module_processor = {}

    @classmethod
    def register_processor(cls, framework, task, processor_class):
        key = (framework, task)
        cls._data_processor[key] = processor_class

    @classmethod
    def register_module_processor(cls, framework, processor_class):
        cls._module_processor[framework] = processor_class

    @classmethod
    def get_module_processor(cls, framework):
        processor_class = cls._module_processor.get(framework)
        if not processor_class:
            raise ValueError(f"ModuleProcessor not found for framework: {framework}")
        return processor_class

    @classmethod
    def create_processor(cls, config, data_writer):
        cls.register_processors(config.framework)
        task = Const.KERNEL_DUMP if config.level == "L2" else config.task
        key = (config.framework, task)
        bench_path = getattr(config, "bench_path", None)
        processor_class = None
        if config.task == Const.TENSOR and bench_path is not None and not isinstance(bench_path, str):
            logger.error_log_with_exp(
                "bench_path is invalid, it should be a string", MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
            )
        elif config.task == Const.TENSOR and bench_path is not None and isinstance(bench_path, str):
            check_file_or_directory_path(bench_path, True)
            processor_class = cls._data_processor.get(("pytorch", Const.DIFF_CHECK))
        else:
            processor_class = cls._data_processor.get(key)
        if not processor_class:
            raise ValueError(f"Processor not found for framework: {config.framework}, task: {config.task}")
        return processor_class(config, data_writer)

    @classmethod
    def register_processors(cls, framework):
        if framework == Const.PT_FRAMEWORK:
            from msprobe.core.dump.data_dump.data_processor.pytorch_processor import (
                StatisticsDataProcessor as PytorchStatisticsDataProcessor,
                TensorDataProcessor as PytorchTensorDataProcessor,
                DiffCheckDataProcessor as PytorchDiffCheckDataProcessor,
                KernelDumpDataProcessor as PytorchKernelDumpDataProcessor,
                NanCheckDataProcessor as PytorchNanCheckDataProcessor,
            )
            from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor

            cls.register_processor(Const.PT_FRAMEWORK, Const.STATISTICS, PytorchStatisticsDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.TENSOR, PytorchTensorDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.KERNEL_DUMP, PytorchKernelDumpDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.DIFF_CHECK, PytorchDiffCheckDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.NAN_CHECK, PytorchNanCheckDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.STRUCTURE, BaseDataProcessor)
            cls.register_module_processor(Const.PT_FRAMEWORK, ModuleProcessor)
        elif framework == Const.MS_FRAMEWORK:
            from msprobe.core.dump.data_dump.data_processor.mindspore_processor import (
                StatisticsDataProcessor as MindsporeStatisticsDataProcessor,
                TensorDataProcessor as MindsporeTensorDataProcessor,
            )
            from msprobe.mindspore.dump.cell_processor import CellProcessor

            cls.register_processor(Const.MS_FRAMEWORK, Const.STATISTICS, MindsporeStatisticsDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.TENSOR, MindsporeTensorDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.STRUCTURE, BaseDataProcessor)
            cls.register_module_processor(Const.MS_FRAMEWORK, CellProcessor)
