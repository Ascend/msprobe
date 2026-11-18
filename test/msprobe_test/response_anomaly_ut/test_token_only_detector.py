# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Barrier
import types
import unittest
from unittest.mock import patch
import weakref
import zipfile

import numpy as np

from msprobe.core.common.log import logger
from msprobe.response_anomaly import DetectionResult, ILLDetector, analyze_output_anomaly
from msprobe.response_anomaly._rare_chars import (
    DEFAULT_MODEL_PATH,
    check_read_dir,
    check_read_file,
    load_safe_npz,
)
from msprobe.response_anomaly._repetition import FEATS, RepetitionDetector, SEP, _describe
from msprobe.response_anomaly._vocab import _parse_spec


FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
CORPORA_THINK_END = {"qwen3": 151668, "glm": 154842, "dsv4": None}


def _byte_encoder():
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    rem = [b for b in range(256) if b not in bs]
    return {b: chr(c) for b, c in zip(bs + rem, bs + [256 + i for i in range(len(rem))])}


def _make_bytelevel_spec(extra_pieces=(), special_tokens=("</think>",)):
    be = _byte_encoder()
    vocab = {be[b]: b for b in range(256)}
    nxt = 256
    for piece in extra_pieces:
        if piece not in vocab:
            vocab[piece] = nxt
            nxt += 1
    added = []
    for sp in special_tokens:
        if sp in vocab:
            tid = vocab[sp]
        else:
            tid = nxt
            nxt += 1
        added.append({"id": tid, "content": sp, "special": True})
    return {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": added,
        "decoder": {"type": "ByteLevel"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
    }


def _pools(tables):
    norm = tables["norm"]
    ids = np.arange(len(norm))
    word = ids[
        (norm > 0)
        & tables["letter"]
        & ~tables["marker"]
        & ~tables["send"]
        & (tables["run_char"] < 0)
        & (tables["chars"] >= 3)
        & (tables["chars"] <= 8)
    ]
    marker = ids[tables["marker"] & ~tables["send"]]
    end = ids[tables["send"] & (norm == SEP) & (tables["chars"] <= 2)]
    run = ids[(tables["run_char"] >= 0) & (tables["chars"] >= 8) & (norm == SEP)]
    return word, marker, end, run


def _make_repetition_seq(corpus, kind, seed, tables):
    rng = np.random.default_rng(seed)
    word, marker, end, run = _pools(tables)

    def text(n):
        out = rng.choice(word, n)
        out[rng.random(n) < 1 / 15] = rng.choice(end)
        return out

    pre = text(int(rng.integers(300, 1200)))
    if kind == "random":
        return text(6000)
    if kind.startswith("loop"):
        period = int(kind[4:])
        return np.concatenate([pre, np.tile(text(period), 4000 // period + 3)])[:5000]
    if kind.startswith("noisy"):
        period = int(kind[5:])
        seg, copies = text(period), []
        for _ in range(5000 // period + 2):
            copy = seg.copy()
            copy[rng.integers(0, period)] = rng.choice(word)
            copies.append(copy)
        return np.concatenate([pre, *copies])[:6000]
    if kind == "run":
        return np.concatenate([pre, np.repeat(rng.choice(run), 120), text(500)])
    if kind in ("circling", "restate"):
        words, marks = rng.choice(word, 12, replace=False), rng.choice(marker, 3)
        fresh = iter(rng.permutation(np.setdiff1d(word, words)))
        body = []
        for _ in range(600):
            sentence = list(rng.permutation(words if kind == "circling" else np.r_[words, next(fresh)]))
            sentence.insert(int(rng.integers(0, len(sentence))), rng.choice(marks))
            body += sentence + [end[0]]
        limit = 7000 if kind == "circling" else 9000
        return np.concatenate([pre, np.array(body)])[:limit]
    if kind == "answer":
        draft, seg = text(700), text(60)
        think_end = CORPORA_THINK_END[corpus] if CORPORA_THINK_END[corpus] is not None else end[0]
        return np.concatenate([pre, draft, draft[:300], [think_end], draft, np.tile(seg, 40)])
    raise ValueError(kind)


class TestRareCharDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = os.path.join(FIXTURES_DIR, "rare_chars_golden.json")
        with open(fixture, encoding="utf-8") as f:
            data = json.load(f)
        cls.model_sha256 = data["model_sha256"]
        cls.golden = data["cases"]
        cls._tmp = tempfile.TemporaryDirectory()
        cls.detectors = {}
        cls.tok_dirs = {}
        for corpus, info in data["mini_tokenizers"].items():
            d = os.path.join(cls._tmp.name, corpus)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(info["tokenizer_json"], f)
            cls.tok_dirs[corpus] = d
            te = None if info["think_end_mode"] == "none" else "auto"
            cls.detectors[corpus] = ILLDetector(d, think_end=te)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_model_asset_integrity(self):
        self.assertLess(os.path.getsize(DEFAULT_MODEL_PATH), 500000)
        with open(DEFAULT_MODEL_PATH, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), self.model_sha256)

    def test_frozen_alerts_and_gb_scores(self):
        for case in self.golden:
            with self.subTest(case=case["name"]):
                detector = self.detectors[case["corpus"]]
                tokens = np.asarray(case["tokens"], np.int64)
                for tier in ("intercept", "monitor"):
                    exp_at, exp_reason = case["expected"][tier]
                    res = detector.detect(tokens, tier=tier)
                    self.assertEqual((res.fired_at, res.reason), (exp_at, exp_reason))
                    if exp_at >= 0:
                        self.assertTrue(res.is_ill)
                        self.assertEqual(res.ill_type, 1)
                    else:
                        self.assertFalse(res.is_ill)
                        self.assertEqual(res.ill_type, 0)
                features = detector.rare_detector.record_features(tokens)
                for pos, expected in case["gb_scores"].items():
                    self.assertAlmostEqual(float(features["gb"][int(pos)]), expected, places=12)

    def test_empty_single_last_and_split_utf8(self):
        detector = self.detectors["qwen3"]
        token_bytes = detector.rare_detector.tables["bytes"]
        lone = token_bytes.index(b"\x80")
        space = token_bytes.index(b" ")
        self.assertEqual(detector.detect([]), DetectionResult())
        self.assertEqual(len(detector.rare_detector.record_features([])["gb"]), 0)
        self.assertEqual(detector.detect([0]), DetectionResult())
        self.assertEqual(
            detector.detect([lone]),
            DetectionResult(is_ill=True, ill_type=1, fired_at=1, reason="bad_utf8"),
        )
        self.assertEqual(
            detector.detect([space, lone]),
            DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8"),
        )
        self.assertEqual(
            detector.detect([lone, space]),
            DetectionResult(is_ill=True, ill_type=1, fired_at=1, reason="bad_utf8"),
        )

        split = [token_bytes.index(bytes([b])) for b in "\u4e2d".encode("utf-8")]
        features = detector.rare_detector.record_features(split)
        self.assertFalse(features["bad_utf8"].any())
        self.assertEqual(detector.detect(split), DetectionResult())

    def test_monitor_rules_and_think_boundary(self):
        detector = self.detectors["qwen3"]
        token_bytes = detector.rare_detector.tables["bytes"]
        a = token_bytes.index(b"a")
        han = token_bytes.index("\u4e2d".encode("utf-8"))
        # Repeating [a] * 101 triggers repetition (ill_type=3) in unified ILLDetector; test rare_detector directly here.
        tokens = [a] * 101 + [han]
        self.assertEqual(detector.rare_detector.detect(tokens), (-1, ""))
        self.assertEqual(
            detector.rare_detector.detect(tokens, tier="monitor"),
            (102, "fs<0.01 & fsw<0.05 & caseless_after_latin"),
        )
        self.assertEqual(
            detector.detect(tokens),
            DetectionResult(is_ill=True, ill_type=3, fired_at=102, reason="vllm[32,2000]x3"),
        )

        # Non-repeating Latin prefix ending in a letter hits monitor rule cleanly on ILLDetector.
        tables = detector.rare_detector.tables
        latin_ids = [
            i for i, (cls, last) in enumerate(zip(tables["rcls"], tables["last"]))
            if cls == 1 and last in (1, 2)
        ]
        rng = np.random.default_rng(0)
        non_rep_tokens = rng.choice(latin_ids, size=101).tolist() + [han]
        self.assertEqual(detector.detect(non_rep_tokens), DetectionResult())
        self.assertEqual(
            detector.detect(non_rep_tokens, tier="monitor"),
            DetectionResult(is_ill=True, ill_type=1, fired_at=102, reason="fs<0.01 & fsw<0.05 & caseless_after_latin"),
        )

        han = token_bytes.index("\u4e00".encode("utf-8"))
        space = token_bytes.index(b" ")
        tokens = [detector.think_end] + [a] * 400
        for _ in range(3):
            tokens += [space, han] + [a] * 15
        tokens += [han]
        self.assertEqual(detector.rare_detector.detect(tokens), (-1, ""))
        self.assertEqual(
            detector.rare_detector.detect(tokens, tier="monitor"),
            (453, "fs<0.01 & answer & caseless_after_latin"),
        )

        features = detector.rare_detector.record_features([0, detector.think_end, 0])
        np.testing.assert_array_equal(features["answer"], [False, True, True])
        dsv4_features = self.detectors["dsv4"].rare_detector.record_features([0, 1])
        np.testing.assert_array_equal(dsv4_features["answer"], [True, True])
        self.assertIsNotNone(self.detectors["glm"].think_end)

        with self.assertRaises(ValueError):
            detector.detect([0], tier="unknown")
        with self.assertRaises(ValueError):
            ILLDetector(self.tok_dirs["qwen3"], think_end=True)
        with self.assertRaises(ValueError):
            ILLDetector(self.tok_dirs["qwen3"], think_end=detector.vocab_size)

    def test_out_of_box_tokenizer_json_dir_and_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            spm_vocab = [
                ["<unk>", 0.0],
                ["<s>", 0.0],
                ["</s>", 0.0],
                ["\u2581the", -1.0],
                ["\u2581answer", -1.2],
                ["\u2581is", -1.1],
                [".", -1.5],
                ["\u9192", -9.0],
            ] + [[f"<0x{b:02X}>", -6.0] for b in range(256)]
            spm_spec = {
                "model": {"type": "Unigram", "vocab": spm_vocab, "unk_id": 0},
                "added_tokens": [
                    {"id": 0, "content": "<unk>", "special": True},
                    {"id": 1, "content": "<s>", "special": True},
                    {"id": 2, "content": "</s>", "special": True},
                    {"id": len(spm_vocab), "content": "</think>", "special": True},
                ],
                "decoder": {
                    "type": "Sequence",
                    "decoders": [
                        {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
                        {"type": "ByteFallback"},
                    ],
                },
            }
            model_dir = os.path.join(tmp, "spm_model")
            os.makedirs(model_dir)
            spm_json = os.path.join(model_dir, "tokenizer.json")
            with open(spm_json, "w", encoding="utf-8") as f:
                json.dump(spm_spec, f)

            det_json = ILLDetector(spm_json)
            det_dir = ILLDetector(model_dir)
            self.assertEqual(det_json.think_end, len(spm_vocab))
            self.assertEqual(det_dir.think_end, len(spm_vocab))

            zhong = [8 + b for b in "\u4e2d".encode("utf-8")]
            self.assertEqual(det_dir.detect([3, 4, 5] + zhong + [6]), DetectionResult())
            bad_utf8_res = DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8")
            self.assertEqual(det_dir.detect([3, 8 + 0x80, 4]), bad_utf8_res)
            self.assertEqual(analyze_output_anomaly([3, 8 + 0x80, 4], model_dir), [[True, 1]])
            self.assertEqual(
                analyze_output_anomaly([3, 8 + 0x80, 4], model_dir, return_detail=True),
                [bad_utf8_res],
            )

            seq = [3, 4, 5] * 60 + [len(spm_vocab)] + [3, 4, 5] * 60 + [7, 6]
            res_json = det_json.rare_detector.detect(seq)
            self.assertEqual(res_json[1], "gb>3.85 & letter")
            self.assertEqual(det_dir.rare_detector.detect(seq), res_json)

            class MockBackend:
                @staticmethod
                def to_str():
                    return json.dumps(spm_spec)

            class MockFastTokenizer:
                backend_tokenizer = MockBackend()
                all_special_ids = [0, 1, 2]

            self.assertEqual(ILLDetector(MockFastTokenizer()).rare_detector.detect(seq), res_json)

            class MockTiktoken:
                mergeable_ranks = {b" hello": 0, b" world": 1, b"\x80": 2, "\u9f98".encode("utf-8"): 3}
                _special_tokens = {"<|endoftext|>": 4, "</think>": 5}

            det_tk = ILLDetector(MockTiktoken())
            self.assertEqual(det_tk.think_end, 5)
            self.assertEqual(
                det_tk.detect([0, 2, 1]),
                DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8"),
            )

            class MockVocabTokenizer:
                all_special_ids = [0, 3]

                @staticmethod
                def get_vocab():
                    return {"<pad>": 0, "\u2581hello": 1, "\u2581world": 2, "</think>": 3, "\u9192": 4, ".": 5}

            det_vocab = ILLDetector(MockVocabTokenizer())
            self.assertEqual(det_vocab.think_end, 3)
            seq_v = [1, 2] * 80 + [3] + [1, 2] * 80 + [4, 5]
            self.assertEqual(det_vocab.rare_detector.detect(seq_v)[1], "gb>3.85 & letter")


class TestRepetitionDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detectors = {}
        for corpus, think_end in CORPORA_THINK_END.items():
            npz_path = os.path.join(FIXTURES_DIR, f"{corpus}.npz")
            cls.detectors[corpus] = RepetitionDetector.from_npz(npz_path, think_end=think_end)
        fixture = os.path.join(FIXTURES_DIR, "repetition_golden.json")
        with open(fixture, encoding="utf-8") as f:
            cls.golden = json.load(f)["cases"]

    def test_fixture_npz_sizes(self):
        for corpus in CORPORA_THINK_END:
            path = os.path.join(FIXTURES_DIR, f"{corpus}.npz")
            self.assertLess(os.path.getsize(path), 500000)

    def test_frozen_alerts_and_features(self):
        for case in self.golden:
            with self.subTest(corpus=case["corpus"], kind=case["kind"]):
                detector = self.detectors[case["corpus"]]
                tokens = _make_repetition_seq(case["corpus"], case["kind"], case["seed"], detector.tables)
                self.assertEqual(len(tokens), case["token_count"])
                for tier in ("intercept", "monitor"):
                    self.assertEqual(detector.detect(tokens, tier), tuple(case["expected"][tier]))
                bnd, features = detector.record_features(tokens)
                self.assertEqual(int(bnd[-1]), case["last_boundary"])
                for name, expected in case["features"].items():
                    self.assertEqual(float(features[-1, FEATS.index(name)]), expected)

    def test_empty_single_and_block_boundaries(self):
        detector = self.detectors["qwen3"]
        vocab_size = len(detector.tables["norm"])
        for n, expected in ((0, []), (1, [1]), (63, [63]), (64, [64]), (65, [64, 65])):
            with self.subTest(n=n):
                tokens = np.arange(n, dtype=np.int64) % vocab_size
                bnd, features = detector.record_features(tokens)
                self.assertEqual(bnd.tolist(), expected)
                self.assertEqual(features.shape, (len(expected), len(FEATS)))
                self.assertEqual(detector.detect(tokens), (-1, ""))
        with self.assertRaises(ValueError):
            detector.detect([0], tier="unknown")

    def test_external_npz_and_think_metadata(self):
        source = self.detectors["qwen3"]
        case = next(c for c in self.golden if c["corpus"] == "qwen3" and c["kind"] == "loop40")
        tokens = _make_repetition_seq("qwen3", case["kind"], case["seed"], source.tables)
        expected = tuple(case["expected"]["intercept"])
        data = {k: v for k, v in source.tables.items() if not k.startswith("_")}
        with tempfile.TemporaryDirectory() as tmp:
            plain = os.path.join(tmp, "plain.npz")
            np.savez(plain, **data)
            with self.assertRaisesRegex(ValueError, "think_end"):
                RepetitionDetector.from_npz(plain)
            detector = RepetitionDetector.from_npz(plain, think_end=151668)
            self.assertEqual(detector.detect(tokens), expected)

            described = os.path.join(tmp, "described.npz")
            np.savez(described, **data, chat=np.array(False), think_end=np.array(151668))
            detector = RepetitionDetector.from_npz(described)
            self.assertEqual(detector.think_end, 151668)
            self.assertEqual(detector.detect(tokens), expected)

            chat = os.path.join(tmp, "chat.npz")
            np.savez(chat, **data, chat=np.array(True), think_end=np.array(-1))
            self.assertIsNone(RepetitionDetector.from_npz(chat).think_end)

    def test_invalid_npz_vocab(self):
        data = {k: v for k, v in self.detectors["qwen3"].tables.items() if not k.startswith("_")}
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.npz")
            np.savez(missing, **{k: v for k, v in data.items() if k != "marker"})
            with self.assertRaisesRegex(ValueError, "missing marker"):
                RepetitionDetector.from_npz(missing, think_end=151668)
            invalid = os.path.join(tmp, "invalid.npz")
            np.savez(invalid, **dict(data, chars=np.zeros(1, np.int16)))
            with self.assertRaisesRegex(ValueError, "invalid chars"):
                RepetitionDetector.from_npz(invalid, think_end=151668)

    def test_pure_python_bytelevel_vocab_builder(self):
        spec = _make_bytelevel_spec(extra_pieces=("wait", "----"), special_tokens=("</think>",))
        vocab = spec["model"]["vocab"]
        wait_id = vocab["wait"]
        run_id = vocab["----"]
        think_id = spec["added_tokens"][0]["id"]
        a_id = vocab["a"]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokenizer.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(spec, f)
            det = ILLDetector(path)

        built = det.repetition_detector.tables
        self.assertEqual(set(built), {"norm", "letter", "chars", "ws", "run_char", "marker", "send"})
        self.assertTrue(all(value.dtype.kind in "biuf" for value in built.values()))
        self.assertTrue(built["marker"][wait_id])
        self.assertEqual(int(built["run_char"][run_id]), ord("-"))
        self.assertEqual(int(built["chars"][run_id]), 4)
        self.assertGreater(built["norm"][think_id], 0)
        self.assertFalse(built["send"][think_id])
        self.assertEqual(det.think_end, think_id)

        self.assertEqual(
            det.detect([a_id] * 200),
            DetectionResult(is_ill=True, ill_type=3, fired_at=128, reason="vllm[32,2000]x3"),
        )

    def test_repeated_calls_are_stateless(self):
        detector = self.detectors["qwen3"]
        run = _make_repetition_seq("qwen3", "run", 1, detector.tables)
        random = _make_repetition_seq("qwen3", "random", 1, detector.tables)
        expected = detector.detect(run)
        self.assertEqual(detector.detect(random), (-1, ""))
        self.assertEqual(detector.detect(run), expected)


class TestUnifiedILLDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        spec = _make_bytelevel_spec(extra_pieces=("wait", "----"), special_tokens=("</think>",))
        cls.tok_path = os.path.join(cls._tmp.name, "tokenizer.json")
        with open(cls.tok_path, "w", encoding="utf-8") as f:
            json.dump(spec, f)
        cls.vocab = spec["model"]["vocab"]
        cls.be = _byte_encoder()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_detection_result_defaults(self):
        res = DetectionResult()
        self.assertFalse(res.is_ill)
        self.assertEqual(res.ill_type, 0)
        self.assertEqual(res.fired_at, -1)
        self.assertEqual(res.reason, "")

    def test_earliest_fired_at_and_tie_break(self):
        det = ILLDetector(self.tok_path)
        a_id = self.vocab["a"]
        space_id = self.vocab[self.be[ord(" ")]]
        lone_id = self.vocab[self.be[0x80]]

        # rare_chars fires at 1; repetition fires at 128 -> rare_chars (1) wins.
        rare_first = [lone_id] + [a_id] * 200
        self.assertEqual(
            det.detect(rare_first),
            DetectionResult(is_ill=True, ill_type=1, fired_at=1, reason="bad_utf8"),
        )

        # repetition fires at 128; rare_chars fires at 201 -> repetition (3) wins.
        rep_first = [a_id] * 200 + [lone_id]
        self.assertEqual(
            det.detect(rep_first),
            DetectionResult(is_ill=True, ill_type=3, fired_at=128, reason="vllm[32,2000]x3"),
        )

        # Both fire at exactly 128 -> rare_chars (1) wins tie.
        tie_seq = [a_id] * 96 + [space_id] * 31 + [lone_id]
        self.assertEqual(len(tie_seq), 128)
        self.assertEqual(det.rare_detector.detect(tie_seq), (128, "bad_utf8"))
        self.assertEqual(det.repetition_detector.detect(tie_seq), (128, "vllm[32,2000]x3"))
        self.assertEqual(
            det.detect(tie_seq),
            DetectionResult(is_ill=True, ill_type=1, fired_at=128, reason="bad_utf8"),
        )

    def test_analyze_batch_alias_cache_and_validation(self):
        det1 = ILLDetector(self.tok_path)
        det2 = ILLDetector(self.tok_path)
        self.assertIs(det1.rare_detector.tables, det2.rare_detector.tables)
        self.assertIs(det1.repetition_detector.tables, det2.repetition_detector.tables)

        a_id = self.vocab["a"]
        lone_id = self.vocab[self.be[0x80]]
        clean = [a_id, self.vocab["b"], self.vocab["c"]]
        rep = [a_id] * 200
        rare = [lone_id]

        self.assertEqual(det1.detector(clean), DetectionResult())
        batch = [clean, rare, rep]
        expected_pairs = [[False, 0], [True, 1], [True, 3]]
        expected_details = [
            DetectionResult(),
            DetectionResult(is_ill=True, ill_type=1, fired_at=1, reason="bad_utf8"),
            DetectionResult(is_ill=True, ill_type=3, fired_at=128, reason="vllm[32,2000]x3"),
        ]
        self.assertEqual(det1.run(batch), expected_pairs)
        self.assertEqual(det1.run(batch, return_detail=True), expected_details)
        self.assertEqual(analyze_output_anomaly(batch, self.tok_path), expected_pairs)
        self.assertEqual(analyze_output_anomaly(batch, self.tok_path, return_detail=True), expected_details)
        self.assertEqual(analyze_output_anomaly([], self.tok_path), [[False, 0]])
        self.assertEqual(analyze_output_anomaly([], self.tok_path, return_detail=True), [DetectionResult()])
        self.assertEqual(analyze_output_anomaly([[]], self.tok_path), [[False, 0]])
        self.assertEqual(analyze_output_anomaly([[]], self.tok_path, return_detail=True), [DetectionResult()])
        self.assertEqual(
            analyze_output_anomaly(np.asarray(rep, dtype=np.int64), self.tok_path),
            [[True, 3]],
        )
        self.assertEqual(
            analyze_output_anomaly(np.asarray(rep, dtype=np.int64), self.tok_path, return_detail=True),
            [expected_details[2]],
        )

        with self.assertRaises(ValueError):
            det1.detect([-1])
        with self.assertRaises(ValueError):
            det1.detect([det1.vocab_size])
        with self.assertRaises(ValueError):
            det1.detect([[a_id]])

    def test_tokenizer_fallbacks_configs_and_edge_cases(self):
        det_path = ILLDetector(Path(self.tok_path), think_end=5)
        self.assertEqual(det_path.think_end, 5)

        a_id = self.vocab["a"]
        arr_2d = np.tile([a_id], (2, 200))
        res_2d = analyze_output_anomaly(arr_2d, Path(self.tok_path))
        self.assertEqual(res_2d, [[True, 3], [True, 3]])
        res_2d_detail = analyze_output_anomaly(arr_2d, Path(self.tok_path), return_detail=True)
        self.assertEqual(len(res_2d_detail), 2)
        self.assertTrue(all(r.ill_type == 3 for r in res_2d_detail))

        class DirectTokenizer:
            @staticmethod
            def get_vocab_size():
                return 4

            @staticmethod
            def to_str():
                return json.dumps({
                    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 3}},
                    "added_tokens": [],
                    "decoder": None,
                })

        det_direct = ILLDetector(DirectTokenizer())
        self.assertEqual(det_direct.vocab_size, 4)
        self.assertEqual(det_direct.detect([0, 1, 3]), DetectionResult())

        class SlottedByteLevelVocab:
            __slots__ = ()

            @staticmethod
            def get_vocab():
                return {"\u0120hello": 0, "\u0120world": 1}

        self.assertEqual(ILLDetector(SlottedByteLevelVocab()).detect([0, 1]), DetectionResult())

        class PlainVocab:
            @staticmethod
            def get_vocab():
                return {"hello": 0, "world": 1}

        self.assertEqual(ILLDetector(PlainVocab()).detect([0, 1]), DetectionResult())

        class EmptyVocab:
            @staticmethod
            def get_vocab():
                return {}

        with self.assertRaises(ValueError):
            ILLDetector(EmptyVocab())
        with self.assertRaises(ValueError):
            ILLDetector(object())
        with self.assertRaises(ValueError):
            _parse_spec({"model": {"vocab": None}})
        with self.assertRaises(ValueError):
            _parse_spec({"model": {"type": "BPE", "vocab": {"\u4e2d": 0}}, "decoder": {"type": "ByteLevel"}})

        spm_bad_hex = {
            "model": {"type": "Unigram", "vocab": [["<0xZZ>", -1.0], ["\u2581ok", -1.0]]},
            "decoder": {"type": "Sequence", "decoders": [{"type": "Metaspace"}, {"type": "ByteFallback"}]},
        }
        chunks, _, _, _, _ = _parse_spec(spm_bad_hex)
        self.assertEqual(chunks[0], b"<0xZZ>")

        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = os.path.join(tmp, "with_cfg")
            os.makedirs(cfg_dir)
            base_spec = _make_bytelevel_spec(special_tokens=())
            with open(os.path.join(cfg_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(base_spec, f)
            with open(os.path.join(cfg_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "added_tokens_decoder": {
                            "0": {"content": "<pad>", "special": True},
                            "1": {"content": "</think>", "special": True},
                        }
                    },
                    f,
                )
            det_cfg = ILLDetector(cfg_dir)
            self.assertEqual(det_cfg.think_end, 1)

            auto_dir = os.path.join(tmp, "auto_fallback")
            os.makedirs(auto_dir)
            with open(os.path.join(auto_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump({"added_tokens_decoder": {"1": {"content": "</think>", "special": True}}}, f)
            fake_transformers = types.ModuleType("transformers")
            observed_kwargs = {}

            class FakeAutoTokenizer:
                @staticmethod
                def from_pretrained(path, trust_remote_code=True, local_files_only=False):
                    observed_kwargs["trust_remote_code"] = trust_remote_code
                    observed_kwargs["local_files_only"] = local_files_only
                    return PlainVocab()

            fake_transformers.AutoTokenizer = FakeAutoTokenizer
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                det_auto = ILLDetector(auto_dir)
                self.assertEqual(det_auto.think_end, 1)
                self.assertIs(observed_kwargs["trust_remote_code"], False)
                self.assertIs(observed_kwargs["local_files_only"], True)

            custom_code_dir = os.path.join(tmp, "custom_code_fallback")
            os.makedirs(custom_code_dir)

            class RejectCustomCodeAutoTokenizer:
                @staticmethod
                def from_pretrained(path, trust_remote_code=True, local_files_only=False):
                    if not trust_remote_code:
                        raise ValueError("Loading this tokenizer requires trust_remote_code=True.")
                    return PlainVocab()

            fake_transformers.AutoTokenizer = RejectCustomCodeAutoTokenizer
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                with self.assertRaisesRegex(ValueError, "trust_remote_code=False"):
                    ILLDetector(custom_code_dir)

            class OtherValueErrorAutoTokenizer:
                @staticmethod
                def from_pretrained(path, trust_remote_code=True, local_files_only=False):
                    raise ValueError("unrecognized model architecture")

            fake_transformers.AutoTokenizer = OtherValueErrorAutoTokenizer
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                with self.assertRaisesRegex(ValueError, "unrecognized model architecture"):
                    ILLDetector(custom_code_dir)

        custom_cfg = {"components": [{"kind": "RUN", "min_chars": 10, "tier": "monitor"}]}
        det_custom = ILLDetector(self.tok_path, repetition_config=custom_cfg)
        self.assertEqual(det_custom.detect([a_id] * 20), DetectionResult())
        self.assertEqual(det_custom.detect([a_id] * 20, tier="monitor").ill_type, 3)

        with self.assertRaises(ValueError):
            ILLDetector(self.tok_path, repetition_config={"components": [{"kind": "BAD"}]})
        with self.assertRaises(ValueError):
            _describe({"kind": "BAD"})
        with self.assertRaises(ValueError):
            det_path.rare_detector.record_features([[0]])
        with self.assertRaises(ValueError):
            det_path.rare_detector.detect([0], tier="bad")
        with self.assertRaises(ValueError):
            det_path.repetition_detector.record_features([[0]])
        with self.assertRaises(ValueError):
            det_path.repetition_detector.record_features([-1])
        with self.assertRaises(ValueError):
            RepetitionDetector.from_npz(os.path.join(FIXTURES_DIR, "qwen3.npz"), think_end=True)

    def test_path_and_npz_security_checks(self):
        with self.assertRaises(FileNotFoundError):
            ILLDetector("/nonexistent/msprobe/tokenizer.json")
        with self.assertRaises(FileNotFoundError):
            check_read_dir("/nonexistent/msprobe/dir")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                check_read_file(tmp)
            with self.assertRaises(ValueError):
                check_read_dir(self.tok_path)
            with self.assertRaises(ValueError):
                check_read_file(self.tok_path, max_size=4)
            with patch("os.access", return_value=False):
                with self.assertRaises(PermissionError):
                    check_read_file(self.tok_path)
                with self.assertRaises(PermissionError):
                    check_read_dir(tmp)
            long_name = os.path.join(tmp, "x" * 300)
            with patch("os.path.exists", return_value=True), patch("os.path.realpath", return_value=long_name):
                with self.assertRaises(ValueError):
                    check_read_file(long_name)
            long_dir = "/" + "d" * 5000
            with patch("os.path.exists", return_value=True), patch("os.path.realpath", return_value=long_dir):
                with self.assertRaises(ValueError):
                    check_read_dir(long_dir)

            link_path = os.path.join(tmp, "tok_link.json")
            try:
                os.symlink(self.tok_path, link_path)
            except OSError:
                link_path = None
            if link_path is not None:
                with patch.object(logger, "warning") as mock_warn:
                    resolved = check_read_file(link_path)
                    self.assertEqual(resolved, os.path.realpath(self.tok_path))
                    self.assertTrue(mock_warn.called)

            bad_trav = os.path.join(tmp, "trav.npz")
            with zipfile.ZipFile(bad_trav, "w") as zf:
                zf.writestr("../evil.npy", b"1234")
            with self.assertRaises(ValueError):
                load_safe_npz(bad_trav)

            too_many = os.path.join(tmp, "many.npz")
            with zipfile.ZipFile(too_many, "w") as zf:
                for idx in range(70):
                    zf.writestr(f"a{idx}.npy", b"1")
            with self.assertRaises(ValueError):
                load_safe_npz(too_many)

            big_npz = os.path.join(tmp, "big.npz")
            np.savez(big_npz, x=np.zeros(32, dtype=np.int64))
            with self.assertRaises(ValueError):
                load_safe_npz(big_npz, max_size=16)


class TestDualModeCompatibility(unittest.TestCase):
    """验证默认无 logprob 模式与 master 原版 logprob 模式的双模式路由及全量兼容。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        spec = _make_bytelevel_spec(extra_pieces=("wait", "----"), special_tokens=("</think>",))
        cls.model_dir = os.path.join(cls._tmp.name, "qwen3-demo")
        os.makedirs(cls.model_dir, exist_ok=True)
        cls.tok_path = os.path.join(cls.model_dir, "tokenizer.json")
        with open(cls.tok_path, "w", encoding="utf-8") as f:
            json.dump(spec, f)

        spec2 = _make_bytelevel_spec(extra_pieces=("hello",), special_tokens=("</think>",))
        cls.model_dir_2 = os.path.join(cls._tmp.name, "glm-demo")
        os.makedirs(cls.model_dir_2, exist_ok=True)
        with open(os.path.join(cls.model_dir_2, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(spec2, f)

        cls.vocab = spec["model"]["vocab"]
        cls.be = _byte_encoder()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_default_token_only_signatures_and_details(self):
        a_id = self.vocab["a"]
        b_id = self.vocab["b"]
        lone_id = self.vocab[self.be[0x80]]
        clean = [a_id, b_id, self.vocab["c"]]
        rare = [a_id, lone_id]
        rep = [a_id] * 200

        # 1) 1D 与 2D 默认位置调用（不传 topk_logprobs）
        self.assertEqual(analyze_output_anomaly(clean, self.model_dir), [[False, 0]])
        self.assertEqual(analyze_output_anomaly(rare, self.model_dir), [[True, 1]])
        self.assertEqual(analyze_output_anomaly(rep, self.model_dir), [[True, 3]])
        self.assertEqual(
            analyze_output_anomaly([clean, rare, rep], self.model_dir),
            [[False, 0], [True, 1], [True, 3]],
        )

        # 2) return_detail=True 保留 fired_at 与 reason
        details = analyze_output_anomaly([clean, rare, rep], self.model_dir, return_detail=True)
        self.assertEqual(details[0], DetectionResult(is_ill=False, ill_type=0, fired_at=-1, reason=""))
        self.assertEqual(details[1], DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8"))
        self.assertEqual(
            details[2],
            DetectionResult(is_ill=True, ill_type=3, fired_at=128, reason="vllm[32,2000]x3"),
        )

        # 3) 关键字传参（支持 model_configs 与 model_or_tokenizer 别名）
        self.assertEqual(
            analyze_output_anomaly(tokens=rare, model_configs=self.tok_path),
            [[True, 1]],
        )
        self.assertEqual(
            analyze_output_anomaly(tokens=rep, model_or_tokenizer=self.model_dir),
            [[True, 3]],
        )

        # 4) 显式传 None 或 dict model_configs、批量多模型路径
        self.assertEqual(analyze_output_anomaly(None, [rare, rep], [self.model_dir, self.tok_path]), [[True, 1], [True, 3]])
        self.assertEqual(
            analyze_output_anomaly([rare, [0, 1]], [self.model_dir, {"model_path": self.model_dir_2}]),
            [[True, 1], [False, 0]],
        )

    def test_master_logprob_mode_regression_and_mixed_batch(self):
        # 构造覆盖 master 全部分类的 logprob 测试样本：
        # 0: 正常; 1: 生僻字(top1_logp < -6, len < stride); 2: 乱码; 3: 重复; 4: NaN
        seq_normal = list(range(1000, 1200))
        lp_normal = [{1000 + j: -0.1 * j for j in range(20)} for _ in range(len(seq_normal))]

        seq_rare_short = [1000] * 10
        lp_rare_short = [{1000: -8.0, 1001: -9.0} for _ in range(len(seq_rare_short))]

        seq_garbled = [1000] * 128
        lp_garbled = [{1000 + j: -10.0 for j in range(20)} for _ in range(len(seq_garbled))]

        seq_rep = [1000] * 1000
        lp_rep = [{1000: -0.01, 1001: -0.1} for _ in range(len(seq_rep))]

        seq_nan = [1000] * 10
        lp_nan = [{1000: float("nan"), 1001: -0.1} for _ in range(len(seq_nan))]

        # 1) master 经典 3 参数位置调用（包含 Qwen3-30B-A3B 与 None）
        self.assertEqual(
            analyze_output_anomaly(
                [lp_normal, lp_rare_short, lp_garbled, lp_rep, lp_nan],
                [seq_normal, seq_rare_short, seq_garbled, seq_rep, seq_nan],
                ["Qwen3-30B-A3B", None, None, "Qwen3-30B-A3B", None],
            ),
            [[False, 0], [True, 1], [True, 2], [True, 3], [True, 4]],
        )

        # 2) master 关键字调用，以及默认位置 + topk_logprobs 关键字调用
        self.assertEqual(
            analyze_output_anomaly(topk_logprobs=[lp_garbled], tokens=[seq_garbled], model_configs=[None]),
            [[True, 2]],
        )
        self.assertEqual(
            analyze_output_anomaly([seq_nan], [None], topk_logprobs=[lp_nan]),
            [[True, 4]],
        )

        # 3) 空 batch 兼容性
        self.assertEqual(analyze_output_anomaly([], [], []), [])
        self.assertEqual(analyze_output_anomaly([], []), [])

        # 4) 同一 batch 混合路由：第 0 条无 logprob(None) 走纯 Token 模式，第 1 条有 logprob 走 master
        a_id = self.vocab["a"]
        lone_id = self.vocab[self.be[0x80]]
        mixed_res = analyze_output_anomaly(
            [None, lp_garbled, None],
            [[a_id, lone_id], seq_garbled, [a_id] * 200],
            [self.model_dir, "Qwen3-30B-A3B", self.tok_path],
            return_detail=True,
        )
        self.assertEqual(mixed_res[0], DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8"))
        self.assertEqual(mixed_res[1], DetectionResult(is_ill=True, ill_type=2, fired_at=-1, reason=""))
        self.assertEqual(
            mixed_res[2],
            DetectionResult(is_ill=True, ill_type=3, fired_at=128, reason="vllm[32,2000]x3"),
        )

    def test_ill_detector_dual_init_and_routing(self):
        a_id = self.vocab["a"]
        lone_id = self.vocab[self.be[0x80]]
        import msprobe.response_anomaly as ra_pkg

        resp_dir = os.path.dirname(os.path.realpath(ra_pkg.__file__))
        cfg_yaml = os.path.join(resp_dir, "configs", "config.yaml")
        mtype_json = os.path.join(resp_dir, "configs", "mtype_config.json")
        tk2cat_dir = os.path.join(resp_dir, "token2category")

        # 1) 无参构造 + legacy 3 参构造 + yaml 单参构造均可执行双模式
        det_default = ILLDetector()
        det_legacy = ILLDetector(cfg_yaml, mtype_json, tk2cat_dir)
        det_yaml = ILLDetector(cfg_yaml)
        self.assertEqual(det_default.window_size, 128)
        self.assertEqual(det_legacy.window_size, 128)
        self.assertEqual(det_yaml.window_size, 128)

        # 未绑定分词器时，在 detector/run 传入 model_dir 即可动态走 token-only
        self.assertEqual(
            det_legacy.detector([a_id, lone_id], self.model_dir),
            DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8"),
        )
        self.assertEqual(
            det_legacy.run([[a_id] * 200], [self.tok_path]),
            [[True, 3]],
        )

        # 2) 用 model_dir 初始化的实例同样兼容 logprob 调用，且可将逻辑模型名回落至已绑定分词器
        det_bound = ILLDetector(self.model_dir)
        lp_nan = [{1000: float("inf"), 1001: -0.1} for _ in range(10)]
        self.assertEqual(det_bound.detector(lp_nan, [1000] * 10, "Qwen3-30B-A3B").ill_type, 4)
        self.assertEqual(det_bound.detector([a_id, lone_id], "Qwen3-30B-A3B").ill_type, 1)
        self.assertEqual(det_bound.run(None, [[a_id] * 200], ["Qwen3-30B-A3B"]), [[True, 3]])
        self.assertEqual(det_bound.detector([], [a_id, lone_id]).ill_type, 1)

        # 3) 绑定分词器 + 显式 config_path 组合初始化
        det_combo = ILLDetector(self.tok_path, config_path=cfg_yaml)
        self.assertEqual(det_combo.window_size, 128)
        self.assertEqual(det_combo.detect([a_id] * 200).ill_type, 3)

        # 4) 无分词器的 legacy 实例显式传入 [] logprobs 保持 master 空输入语义；传入 None 或省略则报错
        self.assertEqual(det_legacy.detector([], [a_id, lone_id]), DetectionResult())
        with self.assertRaises(ValueError):
            det_legacy.detector([a_id, lone_id])
        with self.assertRaises(ValueError):
            det_legacy.detector(None, [a_id, lone_id], "Qwen3-30B-A3B")

    def test_validation_errors(self):
        det_unbound = ILLDetector()
        with self.assertRaises(ValueError):
            det_unbound.detect([1, 2, 3])
        with self.assertRaises(ValueError):
            analyze_output_anomaly([[1, 2, 3]], "Qwen3-30B-A3B")
        with self.assertRaises(ValueError):
            analyze_output_anomaly([[1, 2, 3]], None)
        with self.assertRaises(FileNotFoundError):
            analyze_output_anomaly([], "/nonexistent/msprobe/tokenizer.json")
        with self.assertRaises(ValueError):
            ILLDetector(None)
        with self.assertRaises(ValueError):
            ILLDetector(think_end=True)

        det_bound = ILLDetector(self.tok_path)
        with self.assertRaises(ValueError):
            det_bound.run([1, 2, 3])
        with self.assertRaises(ValueError):
            det_bound.run([[0, 1], [0, 1]], [self.tok_path, self.tok_path, self.tok_path])
        with self.assertRaises(ValueError):
            det_bound.run([None], [[0, 1], [0, 1]])
        with self.assertRaises(ValueError):
            det_bound.detector([0, 1], tier="invalid")
        with self.assertRaises(ValueError):
            det_bound.run([[0, 1]], tier="invalid")
        with self.assertRaises(ValueError):
            analyze_output_anomaly("123", self.tok_path)

    def test_dynamic_cache_reuse_for_unweakrefable_and_dir_fallback(self):
        from msprobe.response_anomaly import _vocab
        from msprobe.response_anomaly.detector import _DYNAMIC_CACHE_MAX

        class SlottedTokenizer:
            __slots__ = ("tag",)

            def __init__(self, tag="default"):
                self.tag = tag

            @staticmethod
            def get_vocab():
                return {"a": 0, "b": 1}

        _vocab._NONWEAK_OBJ_CACHE.clear()
        tok = SlottedTokenizer()
        det = ILLDetector()
        with patch("msprobe.response_anomaly._vocab._decode_and_build", wraps=_vocab._decode_and_build) as spy_build:
            for _ in range(3):
                self.assertEqual(det.run([[0, 1]], tok), [[False, 0]])
            self.assertEqual(spy_build.call_count, 1)
            self.assertEqual(len(det._dynamic_detectors), 1)

        # Verify bounded LRU eviction when rotating many distinct unweakrefable tokenizers.
        det_lru = ILLDetector()
        for idx in range(_DYNAMIC_CACHE_MAX + 5):
            ephemeral = SlottedTokenizer(tag=f"t{idx}")
            self.assertEqual(det_lru.run([[0, 1]], ephemeral), [[False, 0]])
        self.assertLessEqual(len(det_lru._dynamic_detectors), _DYNAMIC_CACHE_MAX)
        self.assertLessEqual(len(_vocab._NONWEAK_OBJ_CACHE), _vocab._NONWEAK_CACHE_MAX)

        # Directory without tokenizer.json (AutoTokenizer fallback) should hit _BUNDLE_CACHE on repeats.
        with tempfile.TemporaryDirectory() as tmp:
            fallback_dir = os.path.join(tmp, "slow_tok")
            os.makedirs(fallback_dir)
            with open(os.path.join(fallback_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump({"added_tokens_decoder": {"1": {"content": "</think>", "special": True}}}, f)

            fake_transformers = types.ModuleType("transformers")
            load_calls = 0

            class CountingAutoTokenizer:
                @staticmethod
                def from_pretrained(path, trust_remote_code=False, local_files_only=False):
                    nonlocal load_calls
                    load_calls += 1
                    return SlottedTokenizer()

            fake_transformers.AutoTokenizer = CountingAutoTokenizer
            det_dir = ILLDetector()
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                for _ in range(3):
                    self.assertEqual(det_dir.run([[0, 1]], fallback_dir), [[False, 0]])
            self.assertEqual(load_calls, 1)
            self.assertEqual(len(det_dir._dynamic_detectors), 1)

    def test_path_bundle_byte_budget_evicts_lru_and_preserves_live_detectors(self):
        from msprobe.response_anomaly import _vocab

        with patch.dict(os.environ, {"MSPROBE_VOCAB_CACHE_MB": "128"}):
            self.assertEqual(_vocab._default_cache_max_bytes(), 128 * 1024 * 1024)
        with patch.dict(os.environ, {"MSPROBE_VOCAB_CACHE_MB": "invalid"}):
            self.assertEqual(_vocab._default_cache_max_bytes(64), 64 * 1024 * 1024)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            _vocab, "_BUNDLE_CACHE", OrderedDict()
        ), patch.object(_vocab, "_OBJ_CACHE", OrderedDict()), patch.object(
            _vocab, "_NONWEAK_OBJ_CACHE", OrderedDict()
        ):
            paths = [Path(tmp) / f"{name}.json" for name in ("a", "b", "c", "d", "large")]
            spec = Path(self.tok_path).read_text(encoding="utf-8")
            for path in paths[:4]:
                path.write_text(spec, encoding="utf-8")
            paths[4].write_text(json.dumps(_make_bytelevel_spec(extra_pieces=("a" * 4096,))), encoding="utf-8")

            first, second = ILLDetector(paths[0]), ILLDetector(paths[1])
            rt = first.rare_detector.tables
            self.assertEqual((rt["tf"].dtype, rt["lcat_s"].dtype, "cp" in rt), (np.float32, np.int8, False))
            self.assertEqual((len(rt["bytes"][:2]), rt["bytes"][-1]), (2, rt["bytes"][len(rt["bytes"]) - 1]))
            size = _vocab._bundle_size_bytes((rt, first.repetition_detector.tables, first.think_end))
            tokens = [self.vocab["a"], self.vocab[self.be[0x80]]]
            expected = DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8")

            with patch.object(_vocab, "_BUNDLE_CACHE_MAX_BYTES", 2 * size):
                self.assertIs(ILLDetector(paths[0]).rare_detector.tables, first.rare_detector.tables)
                ILLDetector(paths[2])
                self.assertEqual((len(_vocab._BUNDLE_CACHE), second.detect(tokens)), (2, expected))
                rebuilt = ILLDetector(paths[1])
                self.assertIsNot(rebuilt.rare_detector.tables, second.rare_detector.tables)
                self.assertEqual(rebuilt.detect(tokens), expected)
                del first, second, rebuilt
                _vocab._BUNDLE_CACHE.clear()
                gc.collect()

                # 跨多个未绑定 ILLDetector 共享统一进程预算：逐出后旧表立即被 GC 释放，且多线程共享单实例并发切换安全。
                det1, det2, refs = ILLDetector(), ILLDetector(), []
                for det, grp in ((det1, paths[:2]), (det2, paths[2:4])):
                    for p in grp:
                        self.assertEqual(det.detect(tokens, model_config=p), expected)
                        refs.append(weakref.ref(_vocab._get_bundle_cache(next(reversed(_vocab._BUNDLE_CACHE)))[0]))
                gc.collect()
                self.assertEqual([r() is None for r in refs], [True, True, False, False])
                self.assertEqual((len(det1._dynamic_detectors), len(det2._dynamic_detectors)), (0, 2))

                shared = ILLDetector()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(
                        pool.map(
                            lambda tid: [shared.detect(tokens, model_config=paths[(tid + s) % 4]) for s in range(16)],
                            range(4),
                        )
                    )
                self.assertTrue(all(r == expected for batch in results for r in batch))

            with patch.object(_vocab, "_BUNDLE_CACHE_MAX", 2):
                _vocab._BUNDLE_CACHE.clear()
                ILLDetector(paths[0])
                ILLDetector(paths[1])
                live_tok = types.SimpleNamespace(get_vocab=lambda: {"a": 0})
                ILLDetector(live_tok)
                self.assertLessEqual(
                    len(_vocab._BUNDLE_CACHE) + len(_vocab._OBJ_CACHE) + len(_vocab._NONWEAK_OBJ_CACHE), 2
                )

            with patch.object(_vocab, "_BUNDLE_CACHE_MAX_BYTES", size):
                _vocab._BUNDLE_CACHE.clear()
                hot, large = ILLDetector(paths[0]), ILLDetector(paths[4])
                self.assertEqual(large.detect(tokens), expected)
                self.assertIsNot(ILLDetector(paths[4]).rare_detector.tables, large.rare_detector.tables)
                self.assertIs(ILLDetector(paths[0]).rare_detector.tables, hot.rare_detector.tables)
                self.assertEqual(len(_vocab._BUNDLE_CACHE), 1)

    def test_oversized_mutation_and_custom_float64_precision(self):
        from msprobe.response_anomaly import _vocab
        from msprobe.response_anomaly._rare_chars import GB_INPUTS, K

        class SlottedTok:
            __slots__ = ("mergeable_ranks", "_special_tokens")

            def __init__(self, ranks):
                self.mergeable_ranks, self._special_tokens = dict(ranks), {}

        class WeakTok:
            def __init__(self, ranks):
                self.mergeable_ranks, self._special_tokens = dict(ranks), {}

        bad_res = DetectionResult(is_ill=True, ill_type=1, fired_at=1, reason="bad_utf8")
        for cls in (SlottedTok, WeakTok):
            with patch.object(_vocab, "_OBJ_CACHE", OrderedDict()), patch.object(
                _vocab, "_NONWEAK_OBJ_CACHE", OrderedDict()
            ):
                tok = cls({b"a": 0})
                probe = ILLDetector(tok)
                bsize = _vocab._bundle_size_bytes((probe.rare_detector.tables, probe.repetition_detector.tables, None))
                del probe
                with patch.object(_vocab, "_BUNDLE_CACHE_MAX_BYTES", bsize + 64):
                    self.assertEqual(ILLDetector(tok).detect([0]), DetectionResult())
                    tok.mergeable_ranks = {b"a": 0, b"b": 1, b"x" * 8192: 2}
                    self.assertEqual(ILLDetector(tok).detect([0]), DetectionResult())
                    tok.mergeable_ranks = {b"\x80": 0}
                    self.assertEqual(ILLDetector(tok).detect([0]), bad_res)

        with tempfile.TemporaryDirectory() as tmp:
            model_npz = Path(tmp) / "custom_rare.npz"
            np.savez(
                model_npz,
                uni_cp=np.array([ord("a")], dtype=np.int32),
                uni_u=np.array([-1.0], dtype=np.float32),
                uni_prev=np.zeros((1, K), dtype=np.float32),
                uni_next=np.zeros((1, K), dtype=np.float32),
                uni_default_u=np.array(1.0 + 2.0**-25, dtype=np.float64),
                uni_default_c=np.array(-5.0, dtype=np.float32),
                gb_inputs=np.array(GB_INPUTS),
                gb_depth=np.array(1, dtype=np.int64),
                gb_feat=np.array([[GB_INPUTS.index("u")]], dtype=np.int64),
                gb_thr=np.array([[1.0 + 2.0**-26]], dtype=np.float64),
                gb_cat=np.array([[0]], dtype=np.int8),
                gb_mask=np.array([[0]], dtype=np.int64),
                gb_leaf=np.array([[0.0, 1.0]], dtype=np.float64),
                gb_base=np.array(0.0, dtype=np.float64),
            )
            det = ILLDetector(self.tok_path, rare_chars_model_path=str(model_npz))
            self.assertEqual(float(det.rare_detector.record_features([self.vocab[self.be[0x80]]])["gb"][0]), 1.0)

    def test_concurrent_path_bundle_publication(self):
        from msprobe.response_anomaly import _vocab

        barrier = Barrier(2)
        build = _vocab._decode_and_build

        def concurrent_build(*args):
            barrier.wait(timeout=10)
            return build(*args)

        with patch.object(_vocab, "_BUNDLE_CACHE", OrderedDict()), patch.object(
            _vocab, "_decode_and_build", concurrent_build
        ), ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(ILLDetector, (self.tok_path, self.tok_path))
            self.assertIs(first.rare_detector.tables, second.rare_detector.tables)
            self.assertIs(first.repetition_detector.tables, second.repetition_detector.tables)
            self.assertEqual(len(_vocab._BUNDLE_CACHE), 1)

    def test_legacy_mixed_positional_and_keyword_args(self):
        lp_normal = [{1000: -0.1, 1001: -0.2}]
        lp_nan = [{1000: float("nan"), 1001: -0.1}]
        a_id = self.vocab["a"]
        lone_id = self.vocab[self.be[0x80]]

        self.assertEqual(analyze_output_anomaly([lp_normal], tokens=[[1000]], model_configs=[None]), [[False, 0]])
        self.assertEqual(analyze_output_anomaly([lp_nan], tokens=[[1000]], model_configs=[None]), [[True, 4]])
        self.assertEqual(analyze_output_anomaly([lp_nan], tokens=[[1000]]), [[True, 4]])
        self.assertEqual(
            analyze_output_anomaly(None, tokens=[a_id, lone_id], model_configs=self.model_dir),
            [[True, 1]],
        )

        det = ILLDetector()
        self.assertEqual(det.run([lp_nan], tokens=[[1000]], model_configs=[None]), [[True, 4]])
        self.assertEqual(det.run([lp_normal], tokens=[[1000]]), [[False, 0]])
        self.assertEqual(det.detector(lp_nan, tokens=[1000], model_config=None), DetectionResult(is_ill=True, ill_type=4))
        self.assertEqual(
            det.detector(None, tokens=[a_id, lone_id], model_config=self.model_dir),
            DetectionResult(is_ill=True, ill_type=1, fired_at=2, reason="bad_utf8"),
        )

        with self.assertRaises(TypeError):
            det.detector(lp_normal, [1000], tokens=[1000])
        with self.assertRaises(TypeError):
            det.run([lp_normal], [[1000]], tokens=[[1000]])

    def test_cwd_config_does_not_hijack_default_configs(self):
        lp_nan = [{1000: float("nan"), 1001: -0.1} for _ in range(10)]
        a_id = self.vocab["a"]
        lone_id = self.vocab[self.be[0x80]]

        with tempfile.TemporaryDirectory() as tmp:
            app_cfg_dir = os.path.join(tmp, "configs")
            os.makedirs(app_cfg_dir)
            with open(os.path.join(app_cfg_dir, "config.yaml"), "w", encoding="utf-8") as f:
                f.write("application:\n  name: unrelated-service\n")

            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                self.assertEqual(analyze_output_anomaly([lp_nan], [[1000] * 10], [None]), [[True, 4]])
                self.assertEqual(analyze_output_anomaly([a_id, lone_id], self.model_dir), [[True, 1]])
                det_default = ILLDetector()
                self.assertEqual(det_default.window_size, 128)
                det_bound = ILLDetector(self.model_dir)
                self.assertEqual(det_bound.detector(lp_nan, [1000] * 10, None).ill_type, 4)
            finally:
                os.chdir(old_cwd)

    def test_mutated_tokenizer_invalidates_object_cache(self):
        class MutableWeakrefTokenizer:
            def __init__(self):
                self._vocab = {"a": 0, "b": 1}
                self.all_special_ids = []

            def get_vocab_size(self):
                return len(self._vocab)

            def to_str(self):
                return json.dumps({
                    "model": {"type": "BPE", "vocab": self._vocab, "merges": []},
                    "added_tokens": [],
                    "decoder": None,
                })

            def add_tokens(self, tokens):
                for tok in tokens:
                    if tok not in self._vocab:
                        self._vocab[tok] = len(self._vocab)

        class MutableSlottedTokenizer:
            __slots__ = ("_vocab", "all_special_ids")

            def __init__(self):
                self._vocab = {"a": 0, "b": 1}
                self.all_special_ids = []

            def get_vocab(self):
                return dict(self._vocab)

            def __len__(self):
                return len(self._vocab)

            def add_tokens(self, tokens):
                for tok in tokens:
                    if tok not in self._vocab:
                        self._vocab[tok] = len(self._vocab)

        for tok in (MutableWeakrefTokenizer(), MutableSlottedTokenizer()):
            det1 = ILLDetector(tok)
            self.assertEqual(det1.vocab_size, 2)
            tok.add_tokens(["c"])
            det2 = ILLDetector(tok)
            self.assertEqual(det2.vocab_size, 3)
            self.assertEqual(det2.detect([0, 1, 2]), DetectionResult())

            unbound = ILLDetector()
            self.assertEqual(unbound.run([[0, 1, 2]], tok), [[False, 0]])
            tok.add_tokens(["d"])
            self.assertEqual(unbound.run([[0, 1, 2, 3]], tok), [[False, 0]])

    def test_causal_fired_at_and_arbitration_invariance(self):
        det = ILLDetector(self.model_dir)
        a_id = self.vocab["a"]
        space_id = self.vocab[self.be[ord(" ")]]
        lone_id = self.vocab[self.be[0x80]]

        # bad_utf8 must remain invariant when normal trailing tokens are appended.
        res_single = det.detect([lone_id])
        res_with_tail = det.detect([lone_id, space_id, a_id])
        self.assertEqual(res_single, DetectionResult(is_ill=True, ill_type=1, fired_at=1, reason="bad_utf8"))
        self.assertEqual(res_with_tail, res_single)

        # Appending a trailing space after a_id * 127 + [lone_id] must not flip ill_type 1 -> 3.
        seq_128 = [a_id] * 127 + [lone_id]
        seq_129 = seq_128 + [space_id]
        self.assertEqual(
            det.detect(seq_128),
            DetectionResult(is_ill=True, ill_type=1, fired_at=128, reason="bad_utf8"),
        )
        self.assertEqual(
            det.detect(seq_129),
            DetectionResult(is_ill=True, ill_type=1, fired_at=128, reason="bad_utf8"),
        )


if __name__ == "__main__":
    unittest.main()
