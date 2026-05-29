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
import os

try:
    import torch
except ImportError:
    torch_available = False
else:
    torch_available = True

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import save_npy, check_path_before_create, check_file_or_directory_path
from msprobe.core.compare.atb_data_compare import TensorBinFile
from msprobe.core.parse.base import BaseParser


class TensorBinFileParser(BaseParser):
    """
    使用TensorBinFile解析.bin文件的解析器
    """

    def parse(self, dump_path, output_path, parse_type):
        """
        解析.bin文件或目录中的.bin文件

        Args:
            dump_path: 输入文件或目录路径
            output_path: 输出路径
            parse_type: 输出文件类型 (npy/pt)
        """
        if not torch_available:
            logger.error('Unable to parse .bin file without torch. Please install with "pip install torch"')
            raise RuntimeError("torch is required for parsing .bin files")

        if os.path.isfile(dump_path):
            self._parse_single_bin_file(dump_path, output_path, parse_type)
        elif os.path.isdir(dump_path):
            bin_files = self._find_bin_files(dump_path)
            logger.info(f"Found {len(bin_files)} .bin file(s) in directory {dump_path}")
            for bin_file in bin_files:
                self._parse_single_bin_file(bin_file, output_path, parse_type)

    def _find_bin_files(self, directory):
        bin_files = []
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) and item.lower().endswith('.bin'):
                    check_file_or_directory_path(item_path, isdir=False)
                    bin_files.append(item_path)
        except (OSError, PermissionError):
            logger.warning(f"Failed to access directory: {directory}")
        return bin_files

    def _parse_single_bin_file(self, bin_file_path, output_path, parse_type):
        try:
            logger.info(f"Parsing .bin file: {bin_file_path}")
            tensor_bin = TensorBinFile(bin_file_path)

            tensor = tensor_bin.get_data()

            if not tensor_bin.is_valid:
                logger.warning(f"Failed to get tensor data from {bin_file_path}")
                return

            if parse_type == 'npy':
                if tensor.dtype == torch.bfloat16:
                    logger.info("Converting bfloat16 tensor to float32 for numpy save")
                    tensor = tensor.float()
                numpy_data = tensor.numpy()
                output_file = BaseParser.get_output_file_path(bin_file_path, output_path, parse_type)
                save_npy(numpy_data, output_file)
                logger.info(f"Saved tensor to {output_file}")
            elif parse_type == 'pt':
                output_file = BaseParser.get_output_file_path(bin_file_path, output_path, parse_type)
                self._save_tensor_to_pt(tensor, output_file)
                logger.info(f"Saved tensor to {output_file}")

        except Exception as e:
            logger.error(f"Failed to parse .bin file {bin_file_path}: {str(e)}")
            raise

    def _save_tensor_to_pt(self, tensor, filepath):
        if not torch_available:
            raise RuntimeError("torch is required for saving .pt files")
        check_path_before_create(filepath)
        filepath = os.path.realpath(filepath)

        try:
            tensor = tensor.contiguous().detach()
            torch.save(tensor, filepath)
        except Exception as e:
            logger.error(f"Save pt file {filepath} failed: {str(e)}")
            raise RuntimeError(f"Save pt file {filepath} failed") from e
