# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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


__version__ = '26.0.0-alpha.3'

import os
import platform
import shutil
import subprocess
import sys

import setuptools
from wheel.bdist_wheel import bdist_wheel

whl_version = os.getenv('WHL_VERSION')
if whl_version is not None:
    __version__ = whl_version

# 检查操作系统，如果不是Linux则报错
if platform.system() != "Linux":
    raise SystemError("This package only supports Linux platform. {}".format(platform.system()))


def build_frontend(plugin_name):
    """构建前端资源"""
    fe_path = os.path.join("plugins", "tb_graph_ascend", plugin_name, "front")
    failed_message = f"Failed to build fronted of {plugin_name},"

    if not os.path.exists(fe_path):
        raise RuntimeError(f"{failed_message} the fronted path '{fe_path}' is not exist")

    original_cwd = os.getcwd()

    try:
        # 切换到fe目录
        os.chdir(fe_path)

        # 检查package.json是否存在
        if not os.path.exists("package.json"):
            raise RuntimeError(f"{failed_message} file 'package.json' is not exist!")

        # 安装依赖
        install_result = subprocess.run(
            ["npm", "install", "--force"],
            capture_output=True,
            text=True
        )
        if install_result.returncode != 0:
            raise RuntimeError(f"{failed_message} run 'npm install --force' failed!")

        # 执行构建
        build_result = subprocess.run(
            ["npm", "run", "build"],
            capture_output=True,
            text=True
        )
        if build_result.returncode != 0:
            raise RuntimeError(f"{failed_message} run 'npm run build' failed!")
    except Exception as e:
        raise RuntimeError(f"{failed_message} {e}")
    finally:
        # 切换回原始目录
        os.chdir(original_cwd)


def clean_frontend_build(plugin_name_list):
    """清除前端构建产物"""
    fe_path = os.path.join("build", "lib")

    if not os.path.exists(fe_path):
        return True

    # 需要清除的目录和文件
    clean_targets = [os.path.join(fe_path, plugin_name) for plugin_name in plugin_name_list]

    cleaned = False
    for target in clean_targets:
        if os.path.exists(target):
            try:
                # 只包含可能抛出异常的代码
                if os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
            except Exception as e:
                print(f"Warning: Failed to clean {target}: {e}")
                continue

            cleaned = True
        else:
            cleaned = True

    return cleaned


INSTALL_REQUIRED = [
    "wheel",
    "einops",
    "numpy >= 1.23.0",
    "pandas >= 1.3.5",
    "pyyaml",
    "tqdm",
    "openpyxl >= 3.0.6",
    "matplotlib",
    "tensorboard >= 2.11.2",
    "protobuf <= 3.20.2",
    "rich",
    "onnx >= 1.14.0",
    "onnxruntime >= 1.14.1, != 1.16.0",
    "skl2onnx >= 1.14.1",
    "setuptools <= 81.0.0",
    "pytz",
    "psutil"
]

if "--plat-name" in sys.argv or "--python-tag" in sys.argv:
    raise SystemError("Specifying platforms or python version is not supported.")

if platform.system() != "Linux":
    raise SystemError("MindStudio-Probe is only supported on Linux platforms.")

# 扩展模块范围，包括adump和tb_graph_ascend
mod_list_range = {"adump", "tb_graph_ascend", "trend_analyzer", "atb_probe", "aclgraph_dump"}
mod_list = []
for i, arg in enumerate(sys.argv):
    if arg.startswith("--include-mod"):
        if "--no-check" in sys.argv:
            os.environ["INSTALL_WITHOUT_CHECK"] = "1"
            sys.argv.remove("--no-check")
        if arg.startswith("--include-mod="):
            mod_list = arg[len("--include-mod="):].split(',')
            sys.argv.remove(arg)
        elif i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            mod_list = sys.argv[i + 1].split(',')
            sys.argv.remove(sys.argv[i + 1])
            sys.argv.remove(arg)
        mod_list = list(set(mod_list) & mod_list_range)
        break

# 处理包含的模块
with_tb_graph_ascend = False
with_trend_analyzer = False
if mod_list:
    # 如果包含tb_graph_ascend，则构建模型分级可视化前端
    if "tb_graph_ascend" in mod_list:
        print("Building tb_graph_ascend frontend...")
        build_frontend('hierarchy_plugin')
        # 设置包含tb_graph_ascend的标志
        with_tb_graph_ascend = True
    else:
        # 不包含tb_graph_ascend，清除相关构建产物
        clean_success = clean_frontend_build(['hierarchy_plugin'])
        if not clean_success:
            raise RuntimeError("警告: 前端构建产物清理不完整")
        # 根据业务需求决定是否继续
        # 可选：raise BuildError(f"清理失败: {e}")
        with_tb_graph_ascend = False

    # 如果包含trend_analyzer，则构建趋势分析可视化前端
    if "trend_analyzer" in mod_list:
        print("Building trend_analyzer frontend...")
        build_frontend('monvis_plugin')
        # 设置包含trend_analyzer的标志
        with_trend_analyzer = True
    else:
        # 不包含tb_graph_ascend，清除相关构建产物
        clean_success = clean_frontend_build(['trend_analyzer'])
        if not clean_success:
            raise RuntimeError("警告: 前端构建产物清理不完整")
        # 根据业务需求决定是否继续
        # 可选：raise BuildError(f"清理失败: {e}")
        with_trend_analyzer = False

    # 如果包含adump/atb_probe/aclgraph_dump，则进行C++相关的构建
    if "adump" in mod_list or "atb_probe" in mod_list or "aclgraph_dump" in mod_list:
        arch = platform.machine()
        sys.argv.append("--plat-name")
        sys.argv.append(f"linux_{arch}")
        sys.argv.append("--python-tag")
        sys.argv.append(f"cp{sys.version_info.major}{sys.version_info.minor}")
        build_cmd = (f"bash ./build.sh -j16 -a {arch} -v {sys.version_info.major}.{sys.version_info.minor}"
                     f" -m {str(mod_list).replace(' ', '')}")
        p = subprocess.run(build_cmd.split(), shell=False)
        if p.returncode != 0:
            raise RuntimeError(f"Failed to build source({p.returncode})")
else:
    # 如果没有指定任何模块，默认不包含tb_graph_ascend，并清除构建产物
    clean_success = clean_frontend_build(['hierarchy_plugin', 'monvis_plugin'])
    if not clean_success:
        raise RuntimeError("警告: 前端构建产物清理不完整")
    with_tb_graph_ascend = False

# 添加scripts脚本
current_dir = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(current_dir, 'scripts')
dst_path = os.path.join(current_dir, 'python', 'msprobe', 'scripts')
if not os.path.isdir(dst_path):
    shutil.copytree(src_path, dst_path)
else:
    for root, dirs, files in os.walk(src_path):
        target_root = os.path.join(dst_path, root[len(src_path) + 1:])
        for dir_name in dirs:
            os.makedirs(os.path.join(target_root, dir_name), mode=0o750, exist_ok=True)
        for file in files:
            shutil.copy(os.path.join(root, file), os.path.join(target_root, file))

# 只查找python目录下的包（msprobe相关）
packages = setuptools.find_packages(where="python")

# 只有在包含tb_graph_ascend时才添加相关包
if with_tb_graph_ascend:
    packages.append('hierarchy_plugin')

if with_trend_analyzer:
    packages.append('trend_analyzer')

# 检查前端是否已构建，决定entry_points内容
entry_points_dict = {
    'console_scripts': ['msprobe=msprobe.msprobe:main'],
}

tensorboard_plugins = []
if with_tb_graph_ascend:
    tensorboard_plugins.append(
        'graph_ascend = hierarchy_plugin.server.plugin:GraphsPlugin'
    )
if with_trend_analyzer:
    tensorboard_plugins.append(
        'TrendVis = trend_analyzer.server.app:TrendVis'
    )
# 只有在包含tensorboard插件时才注册
if tensorboard_plugins:
    entry_points_dict['tensorboard_plugins'] = tensorboard_plugins

# 构建package_dir和package_data
package_dir_config = {"": "python"}
package_data_config = {}

if with_tb_graph_ascend:
    package_dir_config.update({
        'hierarchy_plugin': 'plugins/tb_graph_ascend/hierarchy_plugin'
    })
    package_data_config['hierarchy_plugin'] = ['server/**/*.py', 'server/**/*.js', 'server/**/*.html']

if with_trend_analyzer:
    package_dir_config.update({
        'trend_analyzer': 'plugins/tb_graph_ascend/monvis_plugin',
    })
    package_data_config['trend_analyzer'] = ['server/**/*.py', 'server/**/*.js', 'server/**/*.html']

setuptools.setup(
    name="mindstudio-probe",
    version=__version__,
    description="Ascend MindStudio Probe Utils",
    long_description="MindStudio-Probe is a set of tools for diagnosing and improving model accuracy on Ascend NPU.",
    url="https://gitcode.com/Ascend/MindStudio-Probe",
    author="Ascend Team",
    author_email="pmail_mindstudio@huawei.com",
    packages=packages,
    package_dir=package_dir_config,
    package_data=package_data_config,
    platforms=["Linux"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRED,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Mulan PSL v2',
    keywords='pytorch msprobe ascend',
    ext_modules=[],
    zip_safe=False,
    entry_points=entry_points_dict,
)
