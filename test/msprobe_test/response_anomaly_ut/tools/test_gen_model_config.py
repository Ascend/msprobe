# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import tempfile
import json
import os
import sys
from unittest.mock import MagicMock, patch
sys.modules['transformers'] = MagicMock()
from msprobe.response_anomaly.tools import gen_model_config


class FakeSlowTokenizer:
    """模拟不带 backend_tokenizer 的慢速 tokenizer（如 TikTokenTokenizer）"""

    MODE_FULL_KWARGS = "full_kwargs"
    MODE_NO_KWARGS = "no_kwargs"
    MODE_RAISE = "raise"

    def __init__(self, vocab, decode_results, mode=MODE_FULL_KWARGS):
        self.vocab = vocab
        self.decode_results = decode_results
        self.mode = mode
        self.decode_calls = []
        self.eos_token_id = None

    @property
    def vocab_size(self):
        return max(self.vocab.values()) + 1

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, token_ids, **kwargs):
        token_id = token_ids[0] if isinstance(token_ids, (list, tuple)) else token_ids
        self.decode_calls.append((token_id, dict(kwargs)))
        if self.mode == self.MODE_RAISE:
            raise RuntimeError("decode failed")
        if self.mode == self.MODE_NO_KWARGS and kwargs:
            raise TypeError("decode() got an unexpected keyword argument")
        return self.decode_results.get(token_id, "")


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


class TestDecodeTokenWithFallback(unittest.TestCase):
    def setUp(self):
        self.vocab = {"hello": 0, "▁world": 1, "▁": 2, "<|special|>": 3}
        self.decode_results = {0: "hello", 1: "world", 2: "", 3: "<|special|>"}

    def _fake(self, mode=FakeSlowTokenizer.MODE_FULL_KWARGS):
        return FakeSlowTokenizer(self.vocab, self.decode_results, mode)

    def test_no_backend_tokenizer(self):
        tok = FakeSlowTokenizer(self.vocab, self.decode_results)
        self.assertIsNone(getattr(tok, "backend_tokenizer", None))

    def test_decode_supports_full_kwargs(self):
        tok = self._fake()
        decoded = gen_model_config.decode_token_with_fallback(tok, 0, "hello")
        self.assertEqual(decoded, "hello")
        self.assertEqual(
            tok.decode_calls,
            [(0, {"skip_special_tokens": False, "clean_up_tokenization_spaces": False})],
        )

    def test_decode_falls_back_when_kwargs_rejected(self):
        tok = self._fake(FakeSlowTokenizer.MODE_NO_KWARGS)
        decoded = gen_model_config.decode_token_with_fallback(tok, 0, "hello")
        self.assertEqual(decoded, "hello")
        self.assertEqual(
            [kw for _, kw in tok.decode_calls],
            [{"skip_special_tokens": False, "clean_up_tokenization_spaces": False}, {}],
        )

    def test_decode_falls_back_to_raw_token_when_always_raises(self):
        tok = self._fake(FakeSlowTokenizer.MODE_RAISE)
        decoded = gen_model_config.decode_token_with_fallback(tok, 3, "<|special|>")
        self.assertEqual(decoded, "<|special|>")
        self.assertEqual(len(tok.decode_calls), 2)

    def test_sentencepiece_strip_signature_restores_leading_space(self):
        tok = self._fake()
        self.assertEqual(gen_model_config.decode_token_with_fallback(tok, 1, "▁world"), " world")

    def test_pure_whitespace_token_restored_from_raw(self):
        tok = self._fake()
        self.assertEqual(gen_model_config.decode_token_with_fallback(tok, 2, "▁"), "▁")

    def test_literal_spiece_token_not_rewritten(self):
        tok = FakeSlowTokenizer({"▁ab": 0}, {0: "▁ab"})
        self.assertEqual(gen_model_config.decode_token_with_fallback(tok, 0, "▁ab"), "▁ab")

    def test_bytes_result_decoded_as_utf8(self):
        tok = FakeSlowTokenizer({"hello": 0}, {0: b"caf\xc3\xa9"})
        self.assertEqual(gen_model_config.decode_token_with_fallback(tok, 0, "hello"), "café")

    def test_empty_raw_token_skips_decode(self):
        tok = self._fake()
        self.assertEqual(gen_model_config.decode_token_with_fallback(tok, 9, ""), "")
        self.assertEqual(tok.decode_calls, [])


class TestMainDecodeCompatibility(unittest.TestCase):
    def test_main_writes_id_to_category_for_slow_tokenizer(self):
        vocab = {"hello": 0, "▁world": 1, "▁": 2, "<|special|>": 3}
        decode_results = {0: "hello", 1: "world", 2: "", 3: "<|special|>"}
        tok = FakeSlowTokenizer(vocab, decode_results)
        saved = []
        args = MagicMock(model_path="D:/fake", model_name="fake_model")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(gen_model_config, "parse_args", return_value=args), \
                 patch.object(gen_model_config.AutoTokenizer, "from_pretrained", return_value=tok), \
                 patch.object(gen_model_config, "save_json", side_effect=lambda p, d, **kw: saved.append((p, d))), \
                 patch.object(gen_model_config, "check_path_exists"), \
                 patch("os.getcwd", return_value=tmp):
                gen_model_config.main()

        self.assertEqual(saved[0][1], {"fake-model": {"bos": None, "eos": None}})
        token2cat_path, id_to_category = saved[1]
        self.assertIn("fake-model_4.json", token2cat_path)
        self.assertEqual(
            id_to_category,
            {0: "english_latin", 1: "english_latin_space", 2: "whitespace", 3: "english_latin"},
        )
