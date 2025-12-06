from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fno-wave-loads",
    version="0.1.0",
    author="Research Team",
    author_email="your.email@example.com",
    description="Fourier Neural Operator for wave-structure load prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasanmahin770-ctrl/global-local-fno-wave-loads",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "xarray>=2022.11.0",
        "netCDF4>=1.6.0",
    ],
)
