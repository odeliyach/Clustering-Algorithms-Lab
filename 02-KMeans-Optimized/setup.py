from setuptools import Extension, setup

# Define the C extension module
clustering_extension = Extension(
    "clustering_engine",  # שם ה-module שיהיה ב-Python
    sources=['ext/clustering.c', 'ext/clustering_module.c'],  # קבצי C
    extra_compile_args=['-Wall', '-Wextra', '-O2']  # compiler flags
)

setup(
    name='kmeans-plus-plus',
    version='1.0.0',
    description='Hybrid K-Means++ Implementation with C Extension',
    author='Odel Iyach',
    packages=['src'],  # Python packages
    ext_modules=[clustering_extension],  # C extensions
    install_requires=['numpy>=1.19.0', 'pandas>=1.1.0'],
    python_requires='>=3.8',
)
