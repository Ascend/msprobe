"""builtin_handlers 单元测试 — 验证各 handler 正确委托给 processor。"""
import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.common.output_postprocess.builtin_handlers import (
    postprocess_by_group_index,
    postprocess_by_group_list,
    extract_valid_len_by_group_index,
    extract_valid_len_by_group_list,
)


class TestBuiltinHandlers(unittest.TestCase):

    @patch("msprobe.core.common.output_postprocess.builtin_handlers.processor._clean_by_group_key")
    def test_postprocess_by_group_index(self, mock_clean):
        kwargs = {"group_index": 0}
        postprocess_by_group_index("api_a", "output", None, kwargs)
        mock_clean.assert_called_once_with("api_a", "group_index", "output", kwargs)

    @patch("msprobe.core.common.output_postprocess.builtin_handlers.processor._clean_by_group_key")
    def test_postprocess_by_group_list(self, mock_clean):
        kwargs = {"group_list": [1, 0, 0]}
        postprocess_by_group_list("api_b", "output", None, kwargs)
        mock_clean.assert_called_once_with("api_b", "group_list", "output", kwargs)

    @patch("msprobe.core.common.output_postprocess.builtin_handlers.processor._extract_valid_len_by_group_key")
    def test_extract_valid_len_by_group_index(self, mock_extract):
        kwargs = {"group_index": 5}
        extract_valid_len_by_group_index("api_c", None, kwargs)
        mock_extract.assert_called_once_with("api_c", "group_index", kwargs)

    @patch("msprobe.core.common.output_postprocess.builtin_handlers.processor._extract_valid_len_by_group_key")
    def test_extract_valid_len_by_group_list(self, mock_extract):
        kwargs = {"group_list": [1, 0, 0]}
        extract_valid_len_by_group_list("api_d", None, kwargs)
        mock_extract.assert_called_once_with("api_d", "group_list", kwargs)


if __name__ == "__main__":
    unittest.main()
