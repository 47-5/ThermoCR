from setuptools import setup, find_packages


VERSION = '1.0'
DESCRIPTION = 'ThermoCR -- A tool for calculating thermodynamic quantities and reaction rate constants based on quantum chemical calculations'

setup(
    name="ThermoCR",
    version=VERSION,
    author="Ruichen Liu",
    author_email="1197748182@qq.com",
    description=DESCRIPTION,
    url='https://github.com/47-5/ThermoCR',

    packages=find_packages(),

    install_requires=['pandas', 'numpy', 'scipy', 'ase', 'openpyxl', 'cclib', 'scikit-learn'],
    python_requires='>3.5',



)