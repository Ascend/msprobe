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

"""MindStudio-compliant command line help rendering.

The public CLI uses one renderer so that help text remains consistent when
commands and arguments are added in different modules. Legacy ``<Required>``,
``<Optional>``, and ``<Mandatory>`` help-text prefixes are removed during
rendering. New arguments must express requiredness through their argparse
configuration (for example, ``required=True``) instead of a help-text prefix.
"""

import argparse
import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class HelpSpec:
    description: str
    usage: Optional[str]
    examples: Tuple[Tuple[str, str], ...]
    output: Tuple[str, ...] = ()
    troubleshooting: Tuple[str, ...] = ()


HELP_SPECS: Dict[str, HelpSpec] = {
    "msprobe": HelpSpec(
        "Inspect, compare, and analyze model data on Ascend systems.",
        "msprobe <command> [options]",
        (("Show help for a command", "msprobe compare -h"),),
    ),
    "msprobe compare": HelpSpec(
        "Compare target-side dump data with golden reference data and report accuracy differences.",
        "msprobe compare --target_path <DIR> --golden_path <DIR> [options]",
        (
            ("Compare two dump directories", "msprobe compare --target_path ./npu_dump --golden_path ./golden_dump"),
            (
                "Write results to a custom directory",
                "msprobe compare --target_path ./npu_dump --golden_path ./golden_dump --output_path ./result",
            ),
        ),
        ("<DIR>/compare_result_{timestamp}.csv (or .xlsx)",),
    ),
    "msprobe acc_check": HelpSpec(
        "Run API accuracy checks using the framework recorded in the API information file.",
        "msprobe acc_check -api_info <FILE> [options]",
        (
            (
                "Show options for the framework recorded in the API information file",
                "msprobe acc_check -api_info ./dump_source/step0/rank0/dump.json -h",
            ),
            ("Run an API accuracy check", "msprobe acc_check -api_info ./dump.json"),
        ),
        (
            "<DIR>/accuracy_checking_result_{timestamp}.csv",
            "<DIR>/accuracy_checking_details_{timestamp}.csv",
        ),
    ),
    "msprobe acc_check pytorch": HelpSpec(
        "Run PyTorch API accuracy checks from an API information file.",
        "msprobe acc_check -api_info <FILE> [options]",
        (("Run a PyTorch API accuracy check", "msprobe acc_check -api_info ./dump.json"),),
        (
            "<DIR>/accuracy_checking_result_{timestamp}.csv",
            "<DIR>/accuracy_checking_details_{timestamp}.csv",
        ),
    ),
    "msprobe acc_check mindspore": HelpSpec(
        "Run MindSpore API accuracy checks from an API information file.",
        "msprobe acc_check -api_info <FILE> [options]",
        (("Run a MindSpore API accuracy check", "msprobe acc_check -api_info ./dump.json"),),
        (
            "<DIR>/accuracy_checking_result_{timestamp}.csv",
            "<DIR>/accuracy_checking_details_{timestamp}.csv",
        ),
    ),
    "msprobe multi_acc_check": HelpSpec(
        "Run API accuracy checks in parallel using the framework recorded in the API information file.",
        "msprobe multi_acc_check -api_info <FILE> [options]",
        (
            (
                "Show options for the framework recorded in the API information file",
                "msprobe multi_acc_check -api_info ./dump_source/step0/rank0/dump.json -h",
            ),
            ("Run parallel API accuracy checks", "msprobe multi_acc_check -api_info ./dump.json"),
        ),
        (
            "<DIR>/accuracy_checking_result_{timestamp}.csv",
            "<DIR>/accuracy_checking_details_{timestamp}.csv",
        ),
    ),
    "msprobe multi_acc_check pytorch": HelpSpec(
        "Run PyTorch API accuracy checks in parallel from an API information file.",
        "msprobe multi_acc_check -api_info <FILE> [options]",
        (("Run parallel PyTorch API accuracy checks", "msprobe multi_acc_check -api_info ./dump.json"),),
        (
            "<DIR>/accuracy_checking_result_{timestamp}.csv",
            "<DIR>/accuracy_checking_details_{timestamp}.csv",
        ),
    ),
    "msprobe multi_acc_check mindspore": HelpSpec(
        "Run MindSpore API accuracy checks in parallel from an API information file.",
        "msprobe multi_acc_check -api_info <FILE> [options]",
        (("Run parallel MindSpore API accuracy checks", "msprobe multi_acc_check -api_info ./dump.json"),),
        (
            "<DIR>/accuracy_checking_result_{timestamp}.csv",
            "<DIR>/accuracy_checking_details_{timestamp}.csv",
        ),
    ),
    "msprobe merge_result": HelpSpec(
        "Merge distributed accuracy comparison results into one result set.",
        "msprobe merge_result --input_dir <DIR> --output_dir <DIR> --config-path <FILE> [options]",
        (
            (
                "Merge comparison results",
                "msprobe merge_result --input_dir ./results --output_dir ./merged --config-path ./merge.yaml",
            ),
        ),
        ("<DIR>/multi_ranks_compare_merge_{timestamp}.xlsx",),
    ),
    "msprobe overflow_check": HelpSpec(
        "Analyze dumped tensor data and report overflow information.",
        "msprobe overflow_check --input_path <DIR> [options]",
        (("Analyze one dump step", "msprobe overflow_check --input_path ./dump/step_0"),),
        ("<DIR>/anomaly_analyze_{timestamp}.json",),
    ),
    "msprobe config_check": HelpSpec(
        "Collect, compare, or verify training configuration data.",
        "msprobe config_check (-d [<FILE> ...] | -c <FILE_OR_DIR> <FILE_OR_DIR> | "
        "-vc <NPU_LOG> <BENCH_LOG> | -vv [<BENCH_CONFIG>] <TGT_LOG> | "
        "-sc <NPU_LOG> <BENCH_LOG>) [-o <FILE_OR_DIR>]",
        (
            ("Collect the current training configuration", "msprobe config_check -d ./train.sh"),
            (
                "Compare configurations collected from two environments",
                "msprobe config_check -c ./bench.zip ./cmp.zip",
            ),
            (
                "Compare verl hyperparameters",
                "msprobe config_check -vc ./npu.log ./bench.log",
            ),
            (
                "Verify verl hyperparameters against the default benchmark",
                "msprobe config_check -vv ./tgt.log",
            ),
        ),
        (
            "Dump (-d): <FILE> (default: ./config_check_pack.zip)",
            "Packed-config comparison (-c): <DIR>/result.xlsx (default directory: ./config_check_result)",
            "Checkpoint comparison (-c): <FILE> (default: ./ckpt_similarity.json)",
            "Verl comparison (-vc): <DIR>/NPU_config.json, <DIR>/bench_config.json, and "
            "<DIR>/hyper_params_compare.csv (default directory: ./verl_param_compare_result)",
            "Verl verification (-vv): <DIR>/tgt_config.json and <DIR>/hyper_params_verify.csv "
            "(default directory: ./verl_param_verify_result)",
            "Slime comparison (-sc): <DIR>/NPU_config.json, <DIR>/bench_config.json, and "
            "<DIR>/hyper_params_compare.csv (default directory: ./slime_param_compare_result)",
        ),
    ),
    "msprobe api_precision_compare": HelpSpec(
        "Compare API accuracy result files produced on NPU and GPU devices.",
        "msprobe api_precision_compare -npu <FILE> -gpu <FILE> [options]",
        (
            (
                "Compare two accuracy reports",
                "msprobe api_precision_compare -npu ./npu.csv -gpu ./gpu.csv",
            ),
        ),
        (
            "<DIR>/api_precision_compare_result_{timestamp}.csv",
            "<DIR>/api_precision_compare_details_{timestamp}.csv",
        ),
    ),
    "msprobe graph_visualize": HelpSpec(
        "Build graph visualization data from one or two dump directories.",
        "msprobe graph_visualize --target_path <DIR> --output_path <DIR> [options]",
        (("Build visualization data", "msprobe graph_visualize --target_path ./dump --output_path ./graph"),),
        (
            "Single-graph build: <DIR>/build_{timestamp}.vis.db",
            "Two-graph comparison: <DIR>/compare_{timestamp}.vis.db",
            "Merged graph JSON: <DIR>/step<STEP>/rank<RANK>/{construct.json,dump.json,stack.json}",
        ),
    ),
    "msprobe data2db": HelpSpec(
        "Import dump or monitor data into a SQLite database.",
        "msprobe data2db --db <DIR> --data <DIR> [options]",
        (("Import data with automatic format detection", "msprobe data2db --db ./database --data ./dump"),),
        (
            "Dump data input: <DIR>/dump_data.trend.db",
            "Monitor data input: <DIR>/monitor_data.trend.db",
        ),
    ),
    "msprobe parse": HelpSpec(
        "Parse dumped tensor data into NumPy or PyTorch files.",
        "msprobe parse --dump_path <FILE_OR_DIR> [options]",
        (("Parse dump data as PyTorch files", "msprobe parse --dump_path ./dump --type pt --output_path ./output"),),
        ("<DIR>/{input_basename}.npy or <DIR>/{input_basename}.pt",),
    ),
    "msprobe offline_dump": HelpSpec(
        "Run an offline model and dump intermediate model data for analysis.",
        "msprobe offline_dump --model_path <FILE> [options]",
        (("Dump an offline model", "msprobe offline_dump --model_path ./model.om -o ./output"),),
        (
            "<DIR>/{timestamp}/[{input_name-input_shape}/]dump_data/",
            "<DIR>/{timestamp}/[{input_name-input_shape}/]input/",
            "<DIR>/{timestamp}/[{input_name-input_shape}/]model/",
        ),
    ),
    "msprobe install_deps": HelpSpec(
        "Install optional dependencies required by a selected msprobe mode.",
        "msprobe install_deps --mode <MODE> [options]",
        (("Install offline-mode dependencies", "msprobe install_deps --mode offline"),),
    ),
    "anomaly_processor": HelpSpec(
        "Analyze anomaly detection results and report the earliest high-priority anomalies.",
        None,
        (("Analyze monitor anomalies", "anomaly_processor --data_path ./anomalies"),),
        ("<out_path>/",),
    ),
    "gen_model_config": HelpSpec(
        "Generate response-anomaly model and token-category configuration files.",
        None,
        (("Generate configuration for a local model", "gen_model_config --model-path ./model"),),
        ("../configs/mtype_config.json", "../token2category/{model-name}_{vocab-size}.json"),
    ),
    "acc_check": HelpSpec(
        "Run PyTorch API accuracy checks from an API information file.",
        "acc_check --api_info_file <FILE> [options]",
        (("Run an API accuracy check", "acc_check --api_info_file ./dump.json"),),
        ("<out_path>/accuracy_checking_result_{timestamp}.csv",),
    ),
    "multi_acc_check": HelpSpec(
        "Run PyTorch API accuracy checks in parallel.",
        "multi_acc_check --api_info_file <FILE> [options]",
        (("Run parallel API accuracy checks", "multi_acc_check --api_info_file ./dump.json"),),
        ("<out_path>/accuracy_checking_result_{timestamp}.csv",),
    ),
    "run_overflow_check": HelpSpec(
        "Replay API calls and report tensor overflow conditions.",
        "run_overflow_check --api_info_file <FILE> [options]",
        (("Check APIs for overflow", "run_overflow_check --api_info_file ./dump.json"),),
    ),
    "api_precision_compare": HelpSpec(
        "Compare API accuracy result files produced on NPU and GPU devices.",
        "api_precision_compare -npu <FILE> -gpu <FILE> [options]",
        (("Compare two accuracy reports", "api_precision_compare -npu ./npu.csv -gpu ./gpu.csv"),),
        (
            "<DIR>/api_precision_compare_result_{timestamp}.csv",
            "<DIR>/api_precision_compare_details_{timestamp}.csv",
        ),
    ),
}


_METAVAR_BY_DEST = {
    "api_info": "<FILE>",
    "api_info_file": "<FILE>",
    "config_path": "<FILE>",
    "input_file": "<FILE>",
    "npu_csv_path": "<FILE>",
    "gpu_csv_path": "<FILE>",
    "mapping": "<FILE>",
    "cell_mapping": "<FILE>",
    "api_mapping": "<FILE>",
    "data_mapping": "<FILE>",
    "layer_mapping": "<FILE>",
    "fusion_rule_file": "<FILE>",
    "quant_fusion_rule_file": "<FILE>",
    "close_fusion_rule_file": "<FILE>",
    "dump_path": "<FILE_OR_DIR>",
    "target_path": "<DIR>",
    "golden_path": "<DIR>",
    "input_path": "<DIR>",
    "input_dir": "<DIR>",
    "output_dir": "<DIR>",
    "output_path": "<DIR>",
    "out_path": "<DIR>",
    "db": "<DIR>",
    "data": "<DIR>",
    "mode": "<MODE>",
    "onnx_fusion_switch": "{True,False}",
    "format": "<FORMAT>",
    "rank": "<ID>",
    "device": "<ID>",
    "device_id": "<ID>",
    "step": "<N>",
    "process_num": "<N>",
    "num_splits": "<N>",
}

_LABEL_RE = re.compile(r"^\s*[<\[]\s*(?:required|optional|mandatory)[^>\]]*[>\]]\s*[,.:;-]?\s*", re.IGNORECASE)
_PAREN_DEFAULT_RE = re.compile(r"\s*\((?:default\s*[:=]|default is)\s*[^)]*\)\s*", re.IGNORECASE)
_DEFAULT_MARKER_RE = re.compile(r"\s*\(default\)\s*", re.IGNORECASE)
_TEXT_DEFAULT_RE = re.compile(r"\s*(?:the\s+)?default(?:\s+[a-z_-]+){0,3}\s*(?:is|:|=)\s*[^.;]+[.;]?", re.IGNORECASE)


def _normalise_prog(prog: str) -> str:
    words = prog.replace("\\", "/").split()
    if not words:
        return prog
    words[0] = os.path.basename(words[0])
    if words[0].lower().endswith(".py"):
        words[0] = words[0][:-3]
    return " ".join(words)


def _lookup_spec(parser: argparse.ArgumentParser) -> HelpSpec:
    prog = _normalise_prog(parser.prog)
    spec = HELP_SPECS.get(getattr(parser, "help_spec_key", None) or prog)
    if spec:
        return spec
    description = (parser.description or f"Run the {prog} command.").strip()
    return HelpSpec(description, _build_usage(parser), ((f"Show help for {prog}", f"{prog} -h"),))


def _preferred_options(action: argparse.Action) -> Tuple[Optional[str], Optional[str]]:
    short = next((item for item in action.option_strings if item.startswith("-") and not item.startswith("--")), None)
    long_name = next((item for item in action.option_strings if item.startswith("--")), None)
    if long_name is None:
        long_name = next((item for item in action.option_strings if item != short), None)
    return short, long_name


def _is_flag(action: argparse.Action) -> bool:
    return action.nargs == 0


def _metavar(action: argparse.Action) -> str:
    if action.choices is not None and not isinstance(action.choices, range):
        return "{" + ",".join(str(value).lower() for value in action.choices) + "}"
    value = action.metavar
    if isinstance(value, tuple):
        return " ".join(str(item) for item in value)
    if value:
        value = str(value).strip("<>").upper()
        if value not in {"STRING", "VALUE", "ARG", "PATH"}:
            base = f"<{value}>"
        else:
            base = _METAVAR_BY_DEST.get(action.dest, "<NAME>")
    elif action.dest in _METAVAR_BY_DEST:
        base = _METAVAR_BY_DEST[action.dest]
    elif action.type is int:
        base = "<N>"
    elif action.type is float:
        base = "<FLOAT>"
    elif any(token in action.dest for token in ("path", "file", "mapping", "config")):
        base = "<FILE>"
    elif any(token in action.dest for token in ("num", "count", "size", "top")):
        base = "<N>"
    else:
        base = "<NAME>"

    if action.nargs == "+":
        return f"{base} [{base} ...]"
    if action.nargs == "*":
        return f"[{base} ...]"
    if isinstance(action.nargs, int) and action.nargs > 1:
        return " ".join(base for _ in range(action.nargs))
    return base


def _signature(action: argparse.Action) -> Tuple[str, str]:
    if isinstance(action, argparse._SubParsersAction):
        choices = "{" + ",".join(action.choices) + "}"
        return "", f"<command> {choices}"
    short, long_name = _preferred_options(action)
    if not action.option_strings:
        return "", _metavar(action)
    if long_name is None:
        long_name, short = short, None
    if not _is_flag(action):
        long_name = f"{long_name} {_metavar(action)}"
    return f"{short}," if short else "", long_name or ""


def _description(action: argparse.Action, required: bool) -> str:
    if isinstance(action, argparse._SubParsersAction):
        return "Command to run."
    help_text = "" if action.help in (None, argparse.SUPPRESS) else str(action.help)
    help_text = _LABEL_RE.sub("", help_text).strip()
    help_text = _PAREN_DEFAULT_RE.sub(" ", help_text).strip()
    default = action.default
    has_default = default not in (None, "", argparse.SUPPRESS) and not isinstance(action, argparse._HelpAction)
    if has_default:
        help_text = _DEFAULT_MARKER_RE.sub(" ", help_text)
        help_text = _TEXT_DEFAULT_RE.sub(" ", help_text).strip()
    if help_text:
        help_text = help_text[0].upper() + help_text[1:]
    else:
        help_text = "Show help message." if isinstance(action, argparse._HelpAction) else "Command option."
    if not help_text.endswith((".", "!", "?")):
        help_text += "."

    if required or not has_default:
        return help_text
    if _is_flag(action):
        default_text = "on" if bool(default) else "off"
    elif isinstance(default, (list, tuple)):
        default_text = ",".join(str(item) for item in default)
    else:
        default_text = str(default)
    if "default:" not in help_text.lower():
        help_text = f"{help_text} [default: {default_text}]"
    return help_text


def _required(action: argparse.Action, parser: Optional[argparse.ArgumentParser] = None) -> bool:
    if isinstance(action, argparse._SubParsersAction):
        return True
    if action.option_strings:
        if action.required:
            return True
        if parser is not None:
            return any(group.required and action in group._group_actions for group in parser._mutually_exclusive_groups)
        return False
    return action.nargs not in ("?", "*")


def _visible_actions(parser: argparse.ArgumentParser) -> Iterable[argparse.Action]:
    return (action for action in parser._actions if action.help != argparse.SUPPRESS)


def _build_usage(parser: argparse.ArgumentParser) -> str:
    parts = [_normalise_prog(parser.prog)]
    required_actions = [action for action in _visible_actions(parser) if _required(action, parser)]
    for action in required_actions:
        _, long_name = _signature(action)
        parts.append(long_name)
    if any(not _required(action, parser) for action in _visible_actions(parser)):
        parts.append("[options]")
    return " ".join(parts)


def _format_parameter_section(title: str, actions: Sequence[argparse.Action]) -> str:
    if not actions:
        return ""
    rows = [(*_signature(action), _description(action, title == "Required arguments:")) for action in actions]
    first_width = max(3, *(len(row[0]) for row in rows))
    second_width = max(len(row[1]) for row in rows)
    description_column = 2 + first_width + 1 + second_width + 4
    terminal_width = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns)
    lines = [title]
    for first, second, description in rows:
        prefix = f"  {first:<{first_width}} {second:<{second_width}}    "
        wrapped = textwrap.wrap(description, width=max(20, terminal_width - description_column)) or [""]
        lines.append(prefix + wrapped[0])
        lines.extend(" " * description_column + line for line in wrapped[1:])
    return "\n".join(lines)


def _format_command_section(actions: Sequence[argparse.Action]) -> str:
    rows = []
    for action in actions:
        for command, subparser in action.choices.items():
            rows.append((command, _lookup_spec(subparser).description))
    if not rows:
        return ""

    command_width = max(len(command) for command, _ in rows)
    description_column = 2 + command_width + 4
    terminal_width = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns)
    lines = ["Commands:"]
    for command, description in rows:
        prefix = f"  {command:<{command_width}}    "
        wrapped = textwrap.wrap(description, width=max(20, terminal_width - description_column)) or [""]
        lines.append(prefix + wrapped[0])
        lines.extend(" " * description_column + line for line in wrapped[1:])
    return "\n".join(lines)


def format_help(parser: argparse.ArgumentParser) -> str:
    spec = _lookup_spec(parser)
    usage = spec.usage or _build_usage(parser)
    terminal_width = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns)
    actions = list(_visible_actions(parser))
    command_actions = [action for action in actions if isinstance(action, argparse._SubParsersAction)]
    parameter_actions = [action for action in actions if not isinstance(action, argparse._SubParsersAction)]
    required = [action for action in parameter_actions if _required(action, parser)]
    optional = [action for action in parameter_actions if not _required(action, parser)]
    sections = [
        "Description:\n" + "\n".join(f"  {line}" for line in textwrap.wrap(spec.description, width=terminal_width - 4)),
        f"Usage:\n  {usage}",
        _format_command_section(command_actions),
        _format_parameter_section("Required arguments:", required),
        _format_parameter_section("Optional arguments:", optional),
    ]
    example_lines = ["Examples:"]
    for comment, command in spec.examples:
        example_lines.extend((f"  # {comment}", f"  {command}", ""))
    sections.append("\n".join(example_lines).rstrip())
    if spec.output:
        sections.append("Output:\n" + "\n".join(f"  {line}" for line in spec.output))
    if spec.troubleshooting:
        sections.append("Troubleshooting:\n" + "\n".join(f"  - {line}" for line in spec.troubleshooting))
    return "\n\n".join(section for section in sections if section) + "\n"


class MindStudioArgumentParser(argparse.ArgumentParser):
    """Argument parser with the unified MindStudio help layout."""

    def __init__(self, *args, help_spec_key: Optional[str] = None, **kwargs):
        self.help_spec_key = help_spec_key
        super().__init__(*args, **kwargs)

    def format_help(self) -> str:
        return format_help(self)
