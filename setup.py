from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "mattertune.wrappers.utils.graph_partition",
        ["src/mattertune/wrappers/utils/graph_partition.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="mattertune",
    package_dir={"": "src"},              
    packages=find_packages(where="src"), 
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
