# coding=utf-8
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

"""
Function:
Package the python file into so.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np  


# pip install cython setuptools wheel
extensions = [
    Extension(
        "c2lb", 
        ["c2lb.pyx"],  # 需要编译的 .pyx 文件
        include_dirs=[np.get_include()],
        language="c",  # 使用 C 语言级别
        extra_compile_args=[
            "-O2",
            "-D_FORTIFY_SOURCE=2",
            "-fPIC",
            "-fstack-protector-all",
            "-fno-strict-aliasing",
            "-fno-common",
            "-Wextra",
        ],
        extra_link_args=[
            "-Wl,-z,now",
            "-s",
            "-Wl,-z,relro",
            "-Wl,-z,noexecstack",
        ]
    ),
    Extension(
        "speculative_moe",
        ["speculative_moe.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=[
            "-O2",
            "-D_FORTIFY_SOURCE=2",
            "-fPIC",
            "-fstack-protector-all",
            "-fno-strict-aliasing",
            "-fno-common",
            "-Wextra",
        ],
        extra_link_args=[
            "-Wl,-z,now",
            "-s",
            "-Wl,-z,relro",
            "-Wl,-z,noexecstack",
        ]
    ),
    Extension(
        "c2lb_dynamic", 
        ["c2lb_dynamic.pyx"],  # 需要编译的 .pyx 文件
        include_dirs=[np.get_include()],
        language="c",  # 使用 C 语言级别
        extra_compile_args=[
            "-O2",
            "-D_FORTIFY_SOURCE=2",
            "-fPIC",
            "-fstack-protector-all",
            "-fno-strict-aliasing",
            "-fno-common",
            "-Wextra",
        ],
        extra_link_args=[
            "-Wl,-z,now",
            "-s",
            "-Wl,-z,relro",
            "-Wl,-z,noexecstack",
        ]
    ),
    Extension(
        "c2lb_a3", 
        ["c2lb_a3.pyx"],  # 需要编译的 .pyx 文件
        include_dirs=[np.get_include()],
        language="c",  # 使用 C 语言级别
        extra_compile_args=[
            "-O2",
            "-D_FORTIFY_SOURCE=2",
            "-fPIC",
            "-fstack-protector-all",
            "-fno-strict-aliasing",
            "-fno-common",
            "-Wextra",
        ],
        extra_link_args=[
            "-Wl,-z,now",
            "-s",
            "-Wl,-z,relro",
            "-Wl,-z,noexecstack",
        ]
    )
]


setup(
    name='mypackage',
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),  # 设置语言级别
    zip_safe=False,
)
