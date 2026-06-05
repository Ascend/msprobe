#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 进行比对及结果展示
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from msprobe.core.common.file_utils import get_json_contents, write_csv
import torch
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_register import StandardRegistry
from msprobe.pytorch.api_accuracy_checker.precision_standard.absolute_threshold import AbsolutethdCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.benchmark_compare import BenchmarkCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.ulp_compare import UlpCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.binary_consistency import BinaryCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.thousandth_standard import ThousandthStdCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.accumulative_error_compare import AccumulativeErrorCompare
from msprobe.pytorch.api_accuracy_checker.compare.compare_input import CompareInput
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_abs_err, get_max_abs_err, get_rel_err_ratio, \
    cosine_sim, get_rel_err_origin, get_abs_bench_with_eps, compare_bool_tensor
from msprobe.pytorch.api_accuracy_checker.common.config import msCheckerConfig
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import CompareColumn
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import check_dtype_comparable, \
    DETAIL_TEST_ROWS, BENCHMARK_COMPARE_SUPPORT_LIST
from msprobe.pytorch.api_accuracy_checker.common.utils import extract_basic_api_segments
from msprobe.pytorch.common.log import logger
from msprobe.core.common.decorator import recursion_depth_decorator


ResultInfo = namedtuple('ResultInfo', ['full_api_name', 'fwd_success_status', 'bwd_success_status',
                                       'fwd_compare_alg_results', 'bwd_compare_alg_results', 'rank'])

ResultInfoLike = Union[ResultInfo, Tuple]


# Integer / non-floating dtypes that should fall back to error-rate comparison
_INTEGER_DTYPES = frozenset((bool, np.uint8, np.int8, np.int16, np.uint16, np.uint32, np.int32,
                             np.int64, np.uint64))

# Low-precision floating dtypes (use hundredth + thousandth thresholds)
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)

# High-precision floating dtypes (use thousandth + ten-thousandth thresholds)
_HIGH_PRECISION_DTYPES = (torch.float32, torch.float64)


def _ensure_result_info(result: ResultInfoLike) -> ResultInfo:
    """Normalize a 6-tuple or ResultInfo into a ResultInfo namedtuple.

    Backwards-compatible: callers (e.g. ``record_skip_info``) still pass plain tuples.
    """
    if isinstance(result, ResultInfo):
        return result
    return ResultInfo(*result)


def _format_subject_value(item: Any) -> Any:
    """Format a single value in a detail-csv row to its display representation."""
    if isinstance(item, float):
        return "{:.{}f}".format(item, msCheckerConfig.precision)
    return item


def _aggregate_status(status: Union[str, List[str]]) -> str:
    """Reduce a single status or list of statuses to the worst outcome.

    Order of severity: ERROR > WARNING > PASS > others.
    """
    if not isinstance(status, list):
        return status
    if CompareConst.ERROR in status:
        return CompareConst.ERROR
    if CompareConst.WARNING in status:
        return CompareConst.WARNING
    return CompareConst.PASS


class Comparator:
    # consts for result csv
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"
    COLUMN_MESSAGE = "Message"

    # dropout special-case thresholds
    _DROPOUT_MIN_NUMEL = 100
    _DROPOUT_ZEROS_TOLERANCE = 0.1

    # mapping from CompareConst algorithm name -> (imported compare class, factory hint)
    _STANDARD_COMPARE_CLASSES = {
        CompareConst.ABSOLUTE_THRESHOLD: AbsolutethdCompare,
        CompareConst.BINARY_CONSISTENCY: BinaryCompare,
        CompareConst.ULP_COMPARE: UlpCompare,
        CompareConst.THOUSANDTH_STANDARD: ThousandthStdCompare,
        CompareConst.BENCHMARK: BenchmarkCompare,
        CompareConst.ACCUMULATIVE_ERROR_COMPARE: AccumulativeErrorCompare,
    }

    def __init__(self, result_csv_path: str, details_csv_path: str, is_continue_acc_check: bool,
                 stack_info_json_path: Optional[str] = None, config: Optional[Any] = None) -> None:
        """Initialize a :class:`Comparator` and prepare the output CSVs.

        Args:
            result_csv_path: Filesystem path (or path template containing ``{rank}``)
                for the per-API summary CSV. One row is appended per compared API,
                containing ``API name``, ``Forward Test Success``,
                ``Backward Test Success`` and an optional ``Message``.
            details_csv_path: Filesystem path (or template) for the per-output
                detail CSV. One row is appended for every tensor/element that
                goes through the comparison cascade, holding the metrics
                declared in :data:`DETAIL_TEST_ROWS`.
            is_continue_acc_check: When ``True``, skip writing CSV headers so
                results can be appended to an existing run. When ``False``
                (typical first run), the header row is created via
                :meth:`write_csv_title`.
            stack_info_json_path: Optional path to a JSON file produced by the
                data dump stage, mapping full API names to their traceback
                callstacks. The contents are loaded eagerly and attached to
                ``self.stack_info``; used by :meth:`write_summary_csv` to
                attach stack traces to summary rows.
            config: Reserved for future use. Currently accepted for backwards
                compatibility with ``acc_check.Comparator(config=...)`` but
                not stored on the instance.

        Side Effects:
            - Calls :meth:`_register_compare_func` to build the
              :class:`StandardRegistry` of precision-standard compare classes.
            - May call :meth:`write_csv_title` (when ``is_continue_acc_check``
              is ``False``) which creates empty CSV files with header rows.
            - May load the stack-info JSON file from disk.
        """
        self.save_path_str = result_csv_path
        self.detail_save_path_str = details_csv_path
        self.save_path_list = [result_csv_path]
        self.detail_save_path_list = [details_csv_path]

        self.registry = self._register_compare_func()

        if not is_continue_acc_check:
            self.write_csv_title()
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None

    @staticmethod
    def get_path_from_rank(rank: int, path_list: List[str], path_pattern: str) -> str:
        """Resolve the per-rank save path: if there's only one path in ``path_list`` reuse it,
        otherwise format ``path_pattern`` with ``rank``."""
        return path_list[-1] if len(path_list) == 1 else path_pattern.format(rank)

    @staticmethod
    def print_pretest_result():
        logger.info("Successfully completed acc_check/multi_acc_check.")

    @staticmethod
    def _compare_dropout(bench_output: torch.Tensor, device_output: torch.Tensor) -> Tuple[str, int]:
        """Loose comparison for dropout-style APIs: the position of zero entries may differ.

        Returns:
            A tuple of (status, dummy_column_value). The second slot exists only for
            shape compatibility with ``_compare_core_wrapper`` callers and is unused.
        """
        # Bring both tensors onto the same device to avoid host/device mismatch in subtraction.
        if device_output.device != bench_output.device:
            device_output = device_output.to(bench_output.device)
        bench_output = bench_output.detach()
        device_output = device_output.detach()
        tensor_num = bench_output.numel()
        if tensor_num < Comparator._DROPOUT_MIN_NUMEL:
            return CompareConst.PASS, 1
        bench_zeros = int((bench_output == 0).sum())
        device_zeros = int((device_output == 0).sum())
        if abs(bench_zeros - device_zeros) / tensor_num < Comparator._DROPOUT_ZEROS_TOLERANCE:
            return CompareConst.PASS, 1
        return CompareConst.ERROR, 0

    @staticmethod
    def _compare_builtin_type(bench_output: Any, device_output: Any,
                              compare_column: CompareColumn) -> Tuple[str, CompareColumn, str]:
        """Compare builtin (bool/int/float/str) outputs.

        Note:
            ``bench_output`` is expected to be one of (bool, int, float, str) by the
            caller ``_compare_core``. The leading non-primitive guard is defensive:
            if it triggers it means the dispatch in ``_compare_core`` was bypassed
            and a non-primitive type leaked into this helper, in which case we
            conservatively return ``PASS`` to avoid spurious errors.
        """
        if not isinstance(bench_output, (bool, int, float, str)):
            logger.warning("_compare_builtin_type received non-primitive type %s; "
                           "this should be filtered by _compare_core.", type(bench_output))
            return CompareConst.PASS, compare_column, ""
        if bench_output != device_output:
            return CompareConst.ERROR, compare_column, ""
        compare_column.error_rate = 0
        return CompareConst.PASS, compare_column, ""

    @staticmethod
    def _get_acc_check_detail(test_result: ResultInfoLike) -> List[List[Any]]:
        """Build the per-output detail rows that will be written to the detail CSV.

        Args:
            test_result: Either a :class:`ResultInfo` namedtuple or a 6-tuple
                ``(full_api_name, fwd_status, bwd_status, fwd_results, bwd_results, rank)``.

        Returns:
            A list of rows ready to be persisted via ``write_csv``.
        """
        result = _ensure_result_info(test_result)
        subject_prefix = result.full_api_name
        fwd_result = result.fwd_compare_alg_results
        bwd_result = result.bwd_compare_alg_results

        test_rows: List[List[Any]] = []
        for direction, payload in (("forward", fwd_result), ("backward", bwd_result)):
            if not isinstance(payload, list):
                continue
            for i, test_subject in enumerate(payload):
                subject = f"{subject_prefix}.{direction}.output.{i}"
                formatted = [_format_subject_value(item) for item in test_subject]
                test_rows.append([subject, *formatted])
        return test_rows

    def _run_standard_compare(self, standard: str, input_data: CompareInput) -> None:
        """Instantiate the configured standard's compare class and run ``compare()``.

        The mapping from ``standard`` to a compare class is defined in
        :data:`_STANDARD_COMPARE_CLASSES` and registered against
        :class:`StandardRegistry` in :meth:`_register_compare_func`.
        """
        compare_cls = self._STANDARD_COMPARE_CLASSES[standard]
        compare_cls(input_data).compare()

    def write_csv_title(self) -> None:
        """Initialize the summary and detail CSV files with their header rows."""
        summary_test_rows = [
            [self.COLUMN_API_NAME,
             self.COLUMN_FORWARD_SUCCESS,
             self.COLUMN_BACKWARD_SUCCESS,
             self.COLUMN_MESSAGE]
        ]
        for save_path, detail_save_path in zip(self.save_path_list, self.detail_save_path_list):
            if not os.path.exists(save_path):
                write_csv(summary_test_rows, save_path)
            if not os.path.exists(detail_save_path):
                write_csv(DETAIL_TEST_ROWS, detail_save_path)

    @recursion_depth_decorator("compare_core")
    def _compare_core(self, api_name: str, bench_output: Any, device_output: Any,
                      is_fp8: bool) -> Tuple[str, CompareColumn, str]:
        """Dispatch to the appropriate low-level compare routine based on output type.

        Supported types: ``dict`` (recurse over values), ``torch.Tensor``, builtin
        scalars (``bool``/``int``/``float``/``str``), and ``None``. Anything else
        is reported as ``ERROR`` with an "unexpected output type" message.
        """
        compare_column = CompareColumn()
        status: str
        message: str

        if not isinstance(bench_output, type(device_output)):
            status = CompareConst.ERROR
            message = "bench and npu output type is different."
        elif isinstance(bench_output, dict):
            status, compare_column, message = self._compare_dict(api_name, bench_output, device_output, is_fp8)
        elif isinstance(bench_output, torch.Tensor):
            status, compare_column, message = self._compare_torch_tensor(
                api_name, bench_output, device_output, compare_column, is_fp8)
        elif isinstance(bench_output, (bool, int, float, str)):
            compare_column.bench_type = str(type(bench_output))
            compare_column.npu_type = str(type(device_output))
            status, compare_column, message = self._compare_builtin_type(
                bench_output, device_output, compare_column)
        elif bench_output is None:
            status = CompareConst.SKIP
            message = "Bench output is None, skip this test."
        else:
            status = CompareConst.ERROR
            message = f"Unexpected output type in compare_core: {type(bench_output)}"

        return status, compare_column, message

    def _compare_dict(self, api_name: str, bench_output: dict, device_output: dict,
                      is_fp8: bool) -> Tuple[str, CompareColumn, str]:
        """Compare two dict outputs element-by-element; pass-through to ``_compare_core``."""
        b_keys, n_keys = set(bench_output.keys()), set(device_output.keys())
        if b_keys != n_keys:
            return CompareConst.ERROR, CompareColumn(), "bench and npu output dict keys are different."
        return self._compare_core(api_name, list(bench_output.values()),
                                  list(device_output.values()), is_fp8)

    def write_summary_csv(self, test_result: ResultInfoLike) -> None:
        """Append a single summary row to the per-rank summary CSV."""
        result = _ensure_result_info(test_result)
        df_row = [result.full_api_name, result.fwd_success_status, result.bwd_success_status]
        skip_message = self._extract_skip_message(result)
        if skip_message is not None:
            df_row.append(skip_message)
        if self.stack_info:
            df_row.append("\n".join(self.stack_info[result.full_api_name]))
        save_path = self.get_path_from_rank(result.rank, self.save_path_list, self.save_path_str)
        write_csv([df_row], save_path)

    @staticmethod
    def _extract_skip_message(result: ResultInfo) -> Optional[str]:
        """Return the message of whichever side (fwd/bwd) reported ``SKIP`` first, if any."""
        for compare_alg_results, status in (
                (result.fwd_compare_alg_results, result.fwd_success_status),
                (result.bwd_compare_alg_results, result.bwd_success_status)):
            if status == CompareConst.SKIP and isinstance(compare_alg_results, list) and compare_alg_results:
                return compare_alg_results[0][-1]
        return None

    def write_detail_csv(self, test_result: ResultInfoLike) -> None:
        """Append the per-output detail rows to the per-rank detail CSV."""
        result = _ensure_result_info(test_result)
        detail_save_path = self.get_path_from_rank(result.rank,
                                                   self.detail_save_path_list,
                                                   self.detail_save_path_str)
        write_csv(self._get_acc_check_detail(result), detail_save_path)

    def record_results(self, args: ResultInfoLike) -> None:
        """Write both summary and detail CSVs for a single test result."""
        result = _ensure_result_info(args)
        self.write_summary_csv(result)
        self.write_detail_csv(result)


    def compare_output(self, full_api_name: str, data_info: Any) -> Tuple[bool, bool]:
        """Get compare result and write to result and detail csv.

        Returns:
            A 2-tuple ``(fwd_pass, bwd_pass)`` consumed by the caller to decide
            whether to save the offending inputs for offline inspection.
        """
        _, api_name = extract_basic_api_segments(full_api_name)
        if not api_name:
            raise ValueError(f"API name {full_api_name} has not been adapted.")
        bench_output, device_output = data_info.bench_output, data_info.device_output
        bench_grad, device_grad = data_info.bench_grad, data_info.device_grad
        backward_message = data_info.backward_message
        is_fp8 = data_info.is_fp8

        fwd_success_status, fwd_compare_alg_results = self._compare_one_direction(
            full_api_name, api_name, bench_output, device_output, is_fp8)
        bwd_success_status, bwd_compare_alg_results = self._compare_one_direction(
            full_api_name, api_name,
            bench_grad[0] if (bench_grad and device_grad) else None,
            device_grad[0] if (bench_grad and device_grad) else None,
            is_fp8,
            has_grad=bool(bench_grad and device_grad),
        )

        # If the caller reported a backward message, override the bwd result with a SKIP column.
        if backward_message:
            backward_column = CompareColumn()
            bwd_compare_alg_results = [backward_column.to_column_value(CompareConst.SKIP, backward_message)]
            bwd_success_status = CompareConst.SKIP

        result_info = ResultInfo(full_api_name,
                                 fwd_success_status,
                                 bwd_success_status,
                                 fwd_compare_alg_results,
                                 bwd_compare_alg_results,
                                 data_info.rank)
        self.record_results(result_info)
        bwd_pass = bwd_success_status in (CompareConst.PASS, CompareConst.SPACE)
        return fwd_success_status == CompareConst.PASS, bwd_pass

    def _is_dropout_api(self, full_api_name: str) -> bool:
        """Return True if ``full_api_name`` refers to a dropout-family operator."""
        return full_api_name.startswith("dropout.") or ".dropout." in full_api_name

    def _compare_one_direction(self, full_api_name: str, api_name: str,
                               bench_output: Any, device_output: Any,
                               is_fp8: bool, has_grad: bool = True
                               ) -> Tuple[str, List[List[Any]]]:
        """Compare a single forward or backward direction.

        ``has_grad=False`` is used for the backward slot when no gradient is available.
        """
        if not has_grad:
            return CompareConst.SPACE, []
        if self._is_dropout_api(full_api_name):
            status, _ = self._compare_dropout(bench_output, device_output)
            return status, []
        return self._compare_core_wrapper(api_name, bench_output, device_output, is_fp8)

    def _register_compare_func(self) -> StandardRegistry:
        """Register all configured standard compare functions into a :class:`StandardRegistry`."""
        registry = StandardRegistry()
        for standard in self._STANDARD_COMPARE_CLASSES:
            registry.register(standard, self._make_standard_runner(standard))
        return registry

    def _make_standard_runner(self, standard: str) -> Callable[[CompareInput], None]:
        """Return a closure suitable for :meth:`StandardRegistry.register`.

        The returned callable ignores its return value because each standard
        compare class mutates ``input_data.compare_column`` in place.
        """

        def _runner(input_data: CompareInput) -> None:
            self._run_standard_compare(standard, input_data)

        return _runner

    def _compare_core_wrapper(self, api_name: str, bench_output: Any,
                              device_output: Any, is_fp8: bool
                              ) -> Tuple[str, List[List[Any]]]:
        """Dispatch to :meth:`_compare_core` for scalar / sequence / dict outputs.

        Returns:
            A 2-tuple ``(final_status, detailed_rows)`` where ``final_status`` is the
            worst (ERROR > WARNING > PASS) of all per-element results and
            ``detailed_rows`` is one row per compared element.
        """
        statuses, columns, messages = self._collect_per_element_results(
            api_name, bench_output, device_output, is_fp8)
        detailed_rows = [
            columns[i].to_column_value(statuses[i], messages[i])
            for i in range(len(statuses))
        ]
        return _aggregate_status(statuses), detailed_rows

    def _collect_per_element_results(self, api_name: str, bench_output: Any,
                                     device_output: Any, is_fp8: bool
                                     ) -> Tuple[List[str], List[CompareColumn], List[str]]:
        """Run :meth:`_compare_core` over each element of a sequence, or once for a scalar."""
        if isinstance(bench_output, (list, tuple)):
            if len(bench_output) > len(device_output):
                return [CompareConst.ERROR], [CompareColumn()], ["bench and npu output structure is different."]
            paired = zip(bench_output, device_output[:len(bench_output)])
            return self._zip_compare(api_name, paired, is_fp8)
        status, column, message = self._compare_core(api_name, bench_output, device_output, is_fp8)
        return [status], [column], [message]

    def _zip_compare(self, api_name: str, pairs, is_fp8: bool
                     ) -> Tuple[List[str], List[CompareColumn], List[str]]:
        statuses: List[str] = []
        columns: List[CompareColumn] = []
        messages: List[str] = []
        for b_out_i, n_out_i in pairs:
            s, c, m = self._compare_core(api_name, b_out_i, n_out_i, is_fp8)
            statuses.append(s)
            columns.append(c)
            messages.append(m)
        return statuses, columns, messages

    def _compare_torch_tensor(self, api_name: str, bench_output: torch.Tensor,
                              device_output: torch.Tensor, compare_column: CompareColumn,
                              is_fp8: bool) -> Tuple[str, CompareColumn, str]:
        """Compare two ``torch.Tensor`` outputs.

        Steps:
            1. Detach+clone to avoid mutating caller tensors.
            2. Normalize ``bfloat16`` to ``float32`` for downstream numerical compare.
            3. Move to CPU/numpy and validate shapes / dtypes.
            4. Dispatch to the integer-dtype path or the float-dtype path.
        """
        copy_bench_out = bench_output.detach().clone()
        copy_device_output = device_output.detach().clone()
        compare_column.bench_type = str(copy_bench_out.dtype)
        compare_column.npu_type = str(copy_device_output.dtype)
        compare_column.shape = tuple(copy_device_output.shape)

        if copy_bench_out.shape != copy_device_output.shape:
            return (CompareConst.ERROR, compare_column,
                    f"The shape of bench{str(copy_bench_out.shape)} "
                    f"and npu{str(copy_device_output.shape)} not equal.")

        in_dtype = torch.float8_e4m3fn if is_fp8 else torch.float32
        if copy_device_output.dtype == torch.bfloat16:
            copy_bench_out = copy_bench_out.to(torch.float32)
            copy_device_output = copy_device_output.to(torch.float32)

        bench_np = copy_bench_out.cpu().numpy()
        device_np = copy_device_output.cpu().numpy()

        if not check_dtype_comparable(bench_np, device_np):
            return (CompareConst.ERROR, compare_column,
                    f"Bench out dtype is {bench_np.dtype} but "
                    f"npu output dtype is {device_np.dtype}, cannot compare.")

        if bench_np.size == 0:
            return CompareConst.ERROR, compare_column, "There is not bench calculation result."

        if bench_np.dtype in _INTEGER_DTYPES:
            return self._compare_integer_tensor(bench_np, device_np, compare_column)

        return self._compare_float_tensor(
            api_name, bench_np, device_np, compare_column, copy_device_output.dtype, in_dtype)

    @staticmethod
    def _compare_integer_tensor(bench_output: np.ndarray, device_output: np.ndarray,
                                compare_column: CompareColumn) -> Tuple[str, CompareColumn, str]:
        """Integer/bool dtype path: only error-rate is meaningful, no float precision standard."""
        message = (f"Compare algorithm is not supported for {bench_output.dtype} data. "
                   f"Only judged by Error Rate.\n")
        err_rate, status, msg = compare_bool_tensor(bench_output, device_output)
        compare_column.error_rate = err_rate
        return status, compare_column, message + msg

    def _perform_comparison(self, api_name: str, input_data: CompareInput,
                            dtype: torch.dtype, in_dtype: torch.dtype) -> None:
        """Run the registered precision-standard comparison for ``api_name``/``dtype``."""
        comparison_func = self.registry.get_comparison_function(api_name, dtype, in_dtype)
        comparison_func(input_data)

    def _compare_float_tensor(self, api_name: str, bench_output: np.ndarray,
                              device_output: np.ndarray, compare_column: CompareColumn,
                              dtype: torch.dtype, in_dtype: torch.dtype
                              ) -> Tuple[str, CompareColumn, str]:
        """Run the float precision standard cascade for a numpy tensor pair.

        The cascade is:
            1. Run the registered benchmark compare (if dtype supported) — populates
               the standard-specific metrics on ``compare_column``.
            2. Cosine similarity gate.
            3. Max abs error shortcut (PASS).
            4. Precision-tier-specific relative-error gates
               (low-precision: hundredth→thousandth; high-precision: thousandth→ten-thousandth).
        """
        _, abs_bench_with_eps = get_abs_bench_with_eps(bench_output, dtype)
        abs_err = get_abs_err(bench_output, device_output)
        rel_err_origin = get_rel_err_origin(abs_err, abs_bench_with_eps)

        notes: List[str] = []

        # 1) benchmark / precision-standard pass (side-effect: mutates compare_column)
        input_data = CompareInput(bench_output, device_output, compare_column, dtype, rel_err_origin)
        if str(dtype) in BENCHMARK_COMPARE_SUPPORT_LIST:
            self._perform_comparison(api_name, input_data, dtype, in_dtype)
        else:
            notes.append(f"The data type {dtype} is not supported for new precision standard.")

        # 2) cosine-similarity gate
        early = self._check_cosine(bench_output, device_output, compare_column, notes)
        if early is not None:
            return early, compare_column, "\n".join(notes)

        # 3) max-abs-error shortcut
        early = self._check_max_abs_err(abs_err, compare_column, notes)
        if early is not None:
            return early, compare_column, "\n".join(notes)

        # 4) precision-tier-specific gates
        if dtype in _LOW_PRECISION_DTYPES:
            return self._check_low_precision(rel_err_origin, compare_column, notes)
        if dtype in _HIGH_PRECISION_DTYPES:
            return self._check_high_precision(rel_err_origin, compare_column, notes)
        return CompareConst.PASS, compare_column, "\n".join(notes)

    def _check_cosine(self, bench_output: np.ndarray, device_output: np.ndarray,
                      compare_column: CompareColumn, notes: List[str]) -> Optional[str]:
        """Cosine-similarity gate. Returns ``ERROR`` to short-circuit, else ``None``."""
        cos_res, cos_status, msg = cosine_sim(bench_output, device_output)
        compare_column.cosine_sim = cos_res
        if msg:
            notes.append(msg)
        if not cos_status:
            notes.append("Cosine similarity is less than 0.99, consider as error, "
                         "skip other check and set to SPACE.")
            return CompareConst.ERROR
        return None

    @staticmethod
    def _check_max_abs_err(abs_err: np.ndarray, compare_column: CompareColumn,
                           notes: List[str]) -> Optional[str]:
        """Max-abs-error shortcut. Returns ``PASS`` to short-circuit, else ``None``."""
        max_abs_res, max_abs_status = get_max_abs_err(abs_err)
        compare_column.max_abs_err = max_abs_res
        if max_abs_status:
            notes.append("Max abs error is less than 0.001, consider as pass, "
                         "skip other check and set to SPACE.")
            return CompareConst.PASS
        return None

    @staticmethod
    def _check_low_precision(rel_err_origin: np.ndarray, compare_column: CompareColumn,
                             notes: List[str]) -> Tuple[str, CompareColumn, str]:
        """Low-precision (float16/bfloat16) gate: hundredth → thousandth."""
        hundred_res, hundred_status = get_rel_err_ratio(rel_err_origin, CompareConst.HUNDRED_RATIO_THRESHOLD)
        compare_column.rel_err_hundredth = hundred_res
        if not hundred_status:
            notes.append("Relative error is greater than 0.01, consider as error, "
                         "skip other check and set to SPACE.")
            return CompareConst.ERROR, compare_column, "\n".join(notes)

        thousand_res, thousand_status = get_rel_err_ratio(rel_err_origin, CompareConst.THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_thousandth = thousand_res
        if thousand_status:
            notes.append("Relative error is less than 0.001, consider as pass, "
                         "skip other check and set to SPACE.")
            return CompareConst.PASS, compare_column, "\n".join(notes)
        notes.append("Relative error is greater than 0.001, consider as warning, "
                     "skip other check and set to SPACE.")
        return CompareConst.WARNING, compare_column, "\n".join(notes)

    @staticmethod
    def _check_high_precision(rel_err_origin: np.ndarray, compare_column: CompareColumn,
                              notes: List[str]) -> Tuple[str, CompareColumn, str]:
        """High-precision (float32/float64) gate: thousandth → ten-thousandth."""
        thousand_res, thousand_status = get_rel_err_ratio(rel_err_origin, CompareConst.THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_thousandth = thousand_res
        if not thousand_status:
            notes.append("Relative error is greater than 0.001, consider as error, "
                         "skip other check and set to SPACE.")
            return CompareConst.ERROR, compare_column, "\n".join(notes)

        ten_thousand_res, ten_thousand_status = get_rel_err_ratio(
            rel_err_origin, CompareConst.TEN_THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_ten_thousandth = ten_thousand_res
        if not ten_thousand_status:
            notes.append("Relative error is greater than 0.0001, consider as warning, "
                         "skip other check and set to SPACE.")
            return CompareConst.WARNING, compare_column, "\n".join(notes)
        notes.append("Relative error is less than 0.0001, consider as pass.")
        return CompareConst.PASS, compare_column, "\n".join(notes)
