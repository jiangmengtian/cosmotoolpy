from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "mycosmotoolpy.cpseh",
        ["src/mycosmotoolpy/cpower_spectrum_estimator_half.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "mycosmotoolpy.cpaint_half",
        ["src/mycosmotoolpy/cpaint_half.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="mycosmotoolpy",
    packages=["mycosmotoolpy"],
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
