import unittest
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.pytorch.dump.pt_config import parse_json_config, StatisticsConfig


class TestPtConfig(unittest.TestCase):
    def test_parse_json_config(self):
        mock_json_data = {
            "task": "statistics",
            "dump_path": "./dump/",
            "rank": [],
            "step": [],
            "level": "L1",
            "statistics": {
                "scope": [],
                "list": [],
                "data_mode": ["all"],
            }
        }
        with patch("msprobe.pytorch.dump.pt_config.os.path.join", return_value="/path/config.json"), \
                patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config(None, None)
        self.assertEqual(common_config.task, Const.STATISTICS)
        self.assertEqual(task_config.data_mode, [Const.ALL])


class TestStatisticsConfig(unittest.TestCase):

    def setUp(self):
        self.json_config = {}
        self.config = StatisticsConfig(self.json_config)

    def test_check_summary_mode_valid_statistics(self):
        self.config.summary_mode = Const.STATISTICS
        try:
            self.config._check_summary_mode()
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_check_summary_mode_valid_md5(self):
        self.config.summary_mode = Const.MD5
        try:
            self.config._check_summary_mode()
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_check_summary_mode_invalid(self):
        self.config.summary_mode = "invalid_mode"
        with self.assertRaises(Exception) as context:
            self.config._check_summary_mode()
        self.assertIn(str(context.exception), "[msprobe] 无效参数：")

    def test_check_summary_mode_none(self):
        self.config.summary_mode = None
        try:
            self.config._check_summary_mode()
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")
