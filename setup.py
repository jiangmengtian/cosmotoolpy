from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "cosmotoolpy.cpseh",
        ["src/cosmotoolpy/cpower_spectrum_estimator_half.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "cosmotoolpy.cpaint_half",
        ["src/cosmotoolpy/cpaint_half.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="cosmotoolpy",
    packages=["cosmotoolpy"],
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
