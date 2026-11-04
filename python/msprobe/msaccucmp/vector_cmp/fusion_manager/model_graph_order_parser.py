# coding=utf-8
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the License at:
#
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Parse operator display order from an OM graph JSON file."""

from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.file_utils import FileUtils
from dump_parse.dump import handle_op_name


class ModelGraphOrderParser:
    """Extract normalized operator names and display order from an OM graph."""

    @staticmethod
    def normalize_op_name(op_name: str, ffts: bool = False) -> str:
        """Use the target dump's naming mode without changing its matching rules."""
        normalized_name = op_name.replace('/', '_').replace('.', '_')
        # DumpInfo._handle_one_file bypasses handle_op_name in FFTS mode.
        return normalized_name if ffts else handle_op_name(normalized_name, '')

    @staticmethod
    def _raise_invalid(message: str) -> None:
        log.print_error_log(message)
        raise CompareError(CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR, message)

    def __init__(self, model_graph_file_path: str, ffts: bool = False) -> None:
        self.model_graph_file_path = model_graph_file_path
        self.ffts = ffts

    def parse(self) -> dict:
        """Return ``normalized operator name -> display sequence``."""
        graph_object = FileUtils.load_json_file(self.model_graph_file_path)
        if not isinstance(graph_object, dict):
            self._raise_invalid('The model graph JSON root must be an object.')

        graph_list = graph_object.get('graph')
        if not isinstance(graph_list, list):
            self._raise_invalid('The "graph" field in the model graph JSON must be a list.')

        op_name_to_sequence = {}
        normalized_name_to_original_name = {}
        for graph_index, graph_item in enumerate(graph_list):
            if not isinstance(graph_item, dict):
                self._raise_invalid('The graph at index %d must be an object.' % graph_index)
            op_list = graph_item.get('op')
            if not isinstance(op_list, list):
                self._raise_invalid('The "op" field of graph %d must be a list.' % graph_index)
            self._parse_op_list(
                op_list,
                graph_index,
                op_name_to_sequence,
                normalized_name_to_original_name,
            )

        if not op_name_to_sequence:
            self._raise_invalid('The model graph JSON does not contain an operator.')

        return op_name_to_sequence

    def _parse_op_list(
        self,
        op_list: list,
        graph_index: int,
        op_name_to_sequence: dict,
        normalized_name_to_original_name: dict,
    ) -> None:
        for op_index, op_item in enumerate(op_list):
            if not isinstance(op_item, dict):
                self._raise_invalid('The operator at graph %d index %d must be an object.' % (graph_index, op_index))

            op_name = op_item.get('name')
            if not isinstance(op_name, str) or not op_name:
                self._raise_invalid(
                    'The operator name at graph %d index %d must be a non-empty string.' % (graph_index, op_index)
                )
            try:
                normalized_name = self.normalize_op_name(op_name, self.ffts)
            except IndexError as error:
                self._raise_invalid('Operator name %r has an invalid dump naming field: %s.' % (op_name, error))
            if not normalized_name:
                self._raise_invalid('Operator name %r is empty after dump name normalization.' % op_name)
            if normalized_name in normalized_name_to_original_name:
                previous_name = normalized_name_to_original_name.get(normalized_name)
                self._raise_invalid(
                    'Operator names %r and %r collide after normalization to %r.'
                    % (previous_name, op_name, normalized_name)
                )
            normalized_name_to_original_name[normalized_name] = op_name
            # Append each graph as a whole, following its op array positions.
            op_name_to_sequence[normalized_name] = len(op_name_to_sequence)
