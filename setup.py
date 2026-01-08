"""
Setup configuration for RSQSim Monte Carlo PSHA
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="rsqsim-mc-psha",
    version="0.1.0",
    author="Victor Elendu",
    author_email="your.email@bc.edu",
    description="Monte Carlo Probabilistic Seismic Hazard Assessment using RSQSim catalogs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/rsqsim_mc_psha",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "pygmm>=2.0.0",
        "pyproj>=3.0.0",
        "shapely>=1.8.0",
        "psutil>=5.8.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "cartopy>=0.20.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rsqsim-mc=scripts.run_mc:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)