# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
        "speculative_moe_a3", 
        ["speculative_moe_a3.pyx"],  # 需要编译的 .pyx 文件
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
