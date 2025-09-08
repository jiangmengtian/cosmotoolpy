# cosmotoolpy
A tool package for cosmology research.

## Installation
First, clone the repository and enter the project directory:
```bash
git clone https://github.com/jiangmengtian/cosmotoolpy.git
cd mycosmotoolpy
```
Then install the package in editable mode:
```bash
pip install -e .
```
This will automatically build Cython extensions. If you're modifying `.pyx` files and want to rebuild manually:
```bash
python setup.py build_ext --inplace
```