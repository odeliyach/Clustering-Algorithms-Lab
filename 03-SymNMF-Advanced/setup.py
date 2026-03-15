from setuptools import setup, Extension

module = Extension(
    'symnmfmodule',          
    sources=['ext/symnmfmodule.c', 'ext/symnmf.c'],  
)

setup(
    name='symnmfmodule',
    version='1.0',
    description='C extension for SymNMF',
    ext_modules=[module]
)