from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "cosmotoolpy.cpse",
        ["src/cosmotoolpy/cpower_spectrum_estimator.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "cosmotoolpy.cpaint",
        ["src/cosmotoolpy/cpaint.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="cosmotoolpy",
    packages=["cosmotoolpy"],
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
