from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cythonfn5.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
