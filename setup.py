"""Setup script for Multi-Bin Batching."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multi-bin-batching",
    version="0.1.0",
    author="Multi-Bin Batching Contributors",
    description="Multi-Bin Batching for LLM Inference Throughput Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/multi-bin-batching",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "tqdm>=4.65.0",
            "ipython>=8.12.0",
            "jupyter>=1.0.0",
        ],
    },
    tests_require=["pytest>=7.4.0"],
    test_suite="pytest",
)

