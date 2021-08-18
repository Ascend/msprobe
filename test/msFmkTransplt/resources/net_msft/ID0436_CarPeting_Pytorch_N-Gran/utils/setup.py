from distutils.core import setup, Extension
import ascend_function

orca_module = Extension('orca',
                        sources = ['orcamodule.cpp'],
                        extra_compile_args=['-std=c++11'],)

setup (name = 'orca',
       version = '1.0',
       description = 'ORCA motif counting package',
       ext_modules = [orca_module])

