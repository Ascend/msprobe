# -------------------------------------------------------------------------
# Copyright (c) 2025, Huawei Technologies.
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
# --------------------------------------------------------------------------------------------#
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
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='BSD-3',
    keywords='tensorboard graph ascend plugin',
)
