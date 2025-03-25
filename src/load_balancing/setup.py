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
        "c2lb",  # 这是生成的模块名称，不需要加 .so 后缀
        ["c2lb.pyx"],  # 列出需要编译的 .pyx 文件
        include_dirs=[np.get_include()],  # 如果不使用 NumPy 可以移除这一行
        language="python"  # 使用 python 语言级别
    ),
    Extension(
        "speculative_moe",
        ["speculative_moe.pyx"],
        include_dirs=[np.get_include()],
        language="python"
    )
]


setup(
    name='mypackage',
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),  # 设置语言级别
    zip_safe=False,
)
