"""A script to build the C++ code via Cython.

For building only, the user should not be at
the entry point of the project. It should be
in the folder containing the file "setup.py".

Type:
$ python setup.py build_ext --inplace

"""

import numpy
import setuptools
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

if __name__ == '__main__':
    sources = [
        'interface.pyx',
        'c++/source/extracted_hevc_intraprediction.cpp'
    ]
    ext = Extension('interface',
                    sources=sources,
                    language='c++',
                    extra_compile_args=['-std=c++11'])
    setup(ext_modules=cythonize(ext),
          include_dirs=[numpy.get_include()])


