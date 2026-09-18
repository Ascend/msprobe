# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the BSD 3-Clause License.

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="dlopen requires Linux")


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    work = tmp_path_factory.mktemp("cust_op_api_loader")
    # Stub only the NPU logging dependency; compile the actual production loader.
    logging_header = work / "torch_npu/csrc/framework/utils/OpAdapter.h"
    logging_header.parent.mkdir(parents=True)
    logging_header.write_text(
        '#include <cstdio>\n#define ASCEND_LOGW(...) std::fprintf(stderr, __VA_ARGS__)\n'
    )
    source = work / "loader.cpp"
    source.write_text(
        '#include "cust_op_api_loader.h"\n'
        'int main(int argc, char **argv) {\n'
        '    if (argc != 2) return 2;\n'
        '    void *handle = LoadCheckedCustOpApiLib(argv[1]);\n'
        '    if (handle == nullptr) return 1;\n'
        '    dlclose(handle);\n'
        '    return 0;\n'
        '}\n'
    )
    include_dir = Path(__file__).resolve().parents[3] / "ccsrc/nan_check"
    binary = work / "loader"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(work),
         "-I", str(include_dir), str(source), "-ldl", "-o", str(binary)], check=True
    )
    library_source = work / "library.cpp"
    library_source.write_text('extern "C" int test_symbol() { return 7; }\n')
    library = work / "libcust_opapi.so"
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(library_source), "-o", str(library)], check=True
    )
    return binary, library


@pytest.mark.parametrize("mode,expected", [(0o755, 0), (0o775, 1), (0o757, 1)])
def test_library_permissions(loader, tmp_path, mode, expected):
    binary, library = loader
    candidate = tmp_path / library.name
    shutil.copyfile(library, candidate)
    candidate.chmod(mode)
    assert subprocess.run([str(binary), str(candidate)], check=False).returncode == expected


def test_missing_library(loader, tmp_path):
    binary, _ = loader
    candidate = tmp_path / "missing.so"
    assert subprocess.run([str(binary), str(candidate)], check=False).returncode == 1
