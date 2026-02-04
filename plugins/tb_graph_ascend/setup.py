# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# ==============================================================================
import setuptools

VERSION = '8.2.0-alpha'
INSTALL_REQUIRED = ["tensorboard >= 2.11.2"]

setuptools.setup(
    name="tb-graph-ascend",
    version=VERSION,
    description="Model Hierarchical Visualization TensorBoard Plugin",
    long_description="Model Hierarchical Visualization TensorBoard Plugin : \
        https://gitee.com/ascend/mstt/tree/master/plugins/tensorboard-plugins/tb_graph_ascend",
    url="https://gitee.com/ascend/mstt/tree/master/plugins/tensorboard-plugins/tb_graph_ascend",
    author="Ascend Team",
    author_email="pmail_mindstudio@huawei.com",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "graph_ascend": ["hierarchy_plugin/server/static/**"],
        "mon_vis": ["monvis_plugin/server/static/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "graph_ascend = hierarchy_plugin.server.plugin:GraphsPlugin",
            "mon_vis = monvis_plugin.server.app:TrendVis",
        ],
    },
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRED,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Typescript',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Mulan PSL v2',
    keywords='tensorboard graph ascend plugin',
)
