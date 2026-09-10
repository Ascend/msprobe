# Developer Guide

<!-- md-trans-meta sourceCommit=4d1f90c4053f5cf6d083f7bc3060921d6a055bb8 translatedAt=2026-08-11T02:41:10.626Z pushedAt=2026-08-11T02:51:07.141Z -->

This document is intended for MindStudio Probe (msProbe) developers and maintainers, introducing the source code directory, build methods, feature development processes, verification methods after feature changes, and documentation update requirements. This document is primarily written based on the current MindStudio Probe repository and existing documentation, and is applicable to scenarios such as adding new command parameters, extending tool capabilities, adding deliverables, or maintaining software package installation methods.

## 1. msProbe Development Overview

msProbe provides capabilities such as precision data collection, pre-checking, and comparison for AI task execution. Development work generally falls into the following categories:

| Development Object | Typical Content                                                     |
| ------------------ | ------------------------------------------------------------------ |
| Collection         | Precision data collection for training and inference modules       |
| Pre-checking       | `msprobe acc_check`, `msprobe multi_acc_check`                     |
| Comparison         | `msprobe compare`, `msprobe compare -m atb`, `offline_model`, `msprobe graph_visualize`, etc. |
| Overflow detection | `msprobe overflow_check`                                           |
| Extended features  | Pre-training configuration check, training status monitoring, checkpoint comparison, trend visualization, etc. |
| Documentation      | Installation guide, quick start, feature description, data file reference, extended features |

## 2. Code Directory

Based on the current repository materials, the main directories of the msProbe project are as follows:

| Directory                       | Description                     |
| ------------------------------- | ------------------------------- |
| `ccsrc`                         | C/C++ source code directory     |
| `cmake`                         | Stores CMake files for parsing C-related components |
| `docs`                          | Documentation directory         |
| `examples`                      | Stores tool configuration samples |
| `output`                        | Deliverable generation directory |
| `plugins`                       | Main entry point for plugin code |
| `python/msprobe/core`           | Core functional modules of the tool |
| `python/msprobe/infer`          | Inference tool module           |
| `python/msprobe/mindspore`      | MindSpore tool module           |
| `python/msprobe/msaccucmp`      | msaccucmp tool module           |
| `python/msprobe/overflow_check` | Overflow detection module       |
| `python/msprobe/pytorch`        | PyTorch tool module             |
| `python/msprobe/visualization`  | Visualization module            |
| `scripts`                       | Stores installation, uninstallation, and upgrade scripts |
| `test`                          | Test code directory             |
| `docs/zh`                       | Chinese documentation           |

## 3. Development Environment Configuration

### 3.1 Basic Software

| Software Name            | Version Requirement | Purpose                          |
| ------------------------- | ------------------- | -------------------------------- |
| PyCharm (Recommended) / VS Code | No strict requirement | Write and debug Python code    |
| Python                    | 3.8 or later        | Primary development environment  |
| pip                       | Bundled with Python | Install dependencies and local packages |
| conda                     | No strict requirement | Isolate development dependencies |
| wheel                     | Latest stable version | Build whl packages              |
| Git                       | No strict requirement | Pull, manage, and commit code   |

### 3.2 Development Dependencies

Basic dependencies are defined in `docs/requirements.txt`.

The core runtime dependencies include:

- einops
- matplotlib
- numpy
- openpyxl
- pandas
- pyyaml
- rich
- skl2onnx
- tensorboard
- tqdm
- wheel

### 3.3 Recommended Environment Setup

It is recommended to use a virtual environment in the repository root directory for development:

```bash
conda create -n msprobe python=3.10
conda activate msprobe
```

## 4. Obtaining the Code and Building

### 4.1 Obtaining the Code

```bash
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
```

### 4.2 Compiling and Installing the Basic Toolkit

```bash
pip install uv

python3 build.py
cd ./artifacts
pip install ./mindstudio_probe*.whl
```

When compiling the toolkit, you can also select the functional modules to compile by configuring the --include-mod parameter. For details, see *[msProbe Installation Guide](../install_guide/msprobe_install_guide.md)*

After installation, it is recommended to verify immediately:

```bash
which msprobe
msprobe --help
```

## 5. Testing and Verification

The repository provides a unified unit test entry point:

```bash
cd test/msprobe_test
bash run_test.sh
```

- Test data should be placed in the appropriate location under the `test/` directory.
- After running the tests, code coverage reports are generated in the `./report` directory.

## 6. Document Update

After feature development is completed, if the changes affect user usage or output results, the documentation must be updated synchronously.

| Change Type | Documents Requiring Synchronized Update |
| --- | --- |
| Installation, compilation, and upgrade methods | `docs/en/install_guide/msprobe_install_guide.md` |
| Quick start | `docs/en/quick_start` |
| Feature description | `docs/en/user_guide/dump` |
| Performance baseline  | `docs/en/baseline` |
| Case  | `docs/en/best_practices` |
| FAQ | `docs/en/support` |

If new documents, screenshots, or diagrams are added:

1. Place all images in `docs/en/figures`.
2. File names should correspond to the functional semantics.
3. Figure titles, paths, and descriptive text in the main content must be updated synchronously.

## 7. Submission Process Recommendations

1. After completing feature development, perform local installation verification first.
2. Complete at least one round of `UT`, and supplement with `ST` if necessary.
3. If user-visible behavior changes are involved, synchronously supplement the documentation and example commands.
4. If a new analysis capability is added, describe its input data requirements, output files, and applicable scenarios.
