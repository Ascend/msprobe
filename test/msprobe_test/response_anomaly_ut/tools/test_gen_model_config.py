import unittest
import tempfile
import json
import os
import sys
from unittest.mock import MagicMock
sys.modules['transformers'] = MagicMock()
from msprobe.response_anomaly.tools import gen_model_config


class TestClassifyChar(unittest.TestCase):
    def test_chinese_cjk(self):
        self.assertEqual(gen_model_config._classify_char("中"), "cjk")

    def test_latin(self):
        self.assertEqual(gen_model_config._classify_char("a"), "latin")

    def test_digit(self):
        self.assertEqual(gen_model_config._classify_char("0"), "digit")


class TestCategorizeToken(unittest.TestCase):
    def test_chinese_token(self):
        info = gen_model_config.categorize_token(1000, "中文", "中文")
        self.assertEqual(info.category, "chinese_cjk")

    def test_english_token(self):
        info = gen_model_config.categorize_token(1001, "hello", "hello")
        self.assertEqual(info.category, "english_latin")

    def test_number_token(self):
        info = gen_model_config.categorize_token(1002, "123", "123")
        self.assertEqual(info.category, "numbers")


class TestInvertVocab(unittest.TestCase):
    def test_invert_vocab_basic(self):
        vocab = {"hello": 0, "world": 1}
        tokens = gen_model_config.invert_vocab(vocab)
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0], "hello")
        self.assertEqual(tokens[1], "world")


class TestNormalizeName(unittest.TestCase):
    def test_normalize_name_basic(self):
        self.assertEqual(gen_model_config._normalize_name("Qwen-7B"), "qwen-7b")


class TestReadTokenId(unittest.TestCase):
    def test_read_tokenid_valid_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"eos_token_id": 50256, "bos_token_id": 50257}, f)
            temp_path = f.name
        try:
            eos, bos = gen_model_config.read_tokenid(temp_path)
            self.assertEqual(eos, 50256)
            self.assertEqual(bos, 50257)
        finally:
            os.unlink(temp_path)

    def test_read_tokenid_nonexistent_file(self):
        eos, bos = gen_model_config.read_tokenid("/nonexist/path.json")
        self.assertIsNone(eos)
        self.assertIsNone(bos)


class TestScriptlabels(unittest.TestCase):
    def test_cjk_label(self):
        self.assertEqual(gen_model_config.SCRIPT_LABELS["cjk"], "chinese_cjk")

    def test_hiragana_label(self):
        self.assertEqual(
            gen_model_config.SCRIPT_LABELS["hiragana"], "japanese_hiragana"
        )


class TestConstants(unittest.TestCase):
    def test_punct_chars_not_empty(self):
        self.assertGreater(len(gen_model_config.PUNCT_CHARS), 0)

    def test_whitespace_chars_not_empty(self):
        self.assertGreater(len(gen_model_config.WHITESPACE_CHARS), 0)


class TestTokenInfoDataclass(unittest.TestCase):
    def test_token_info_creation(self):
        info = gen_model_config.TokenInfo(
            token_id=1000,
            category="english_latin",
        )
        self.assertEqual(info.token_id, 1000)
        self.assertEqual(info.category, "english_latin")
