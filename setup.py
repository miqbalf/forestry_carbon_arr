"""
Setup script for Forestry Carbon ARR library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Forestry Carbon ARR Library"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="forestry_carbon_arr",
    version="0.1.0",
    author="GIS Carbon AI Team",
    author_email="muh.firdausiqbal@gmail.com",
    description="A library for managing GEE_notebook_Forestry integration in forestry carbon analysis workflows",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/forestry_carbon_arr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gee": [
            "earthengine-api>=0.1.370",
            "geemap>=0.30.0",
            "folium>=0.15.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forestry-carbon-arr=forestry_carbon_arr.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "forestry_carbon_arr": [
            "config/*.json",
            "config/*.yaml",
            "data/*.json",
            "data/*.csv",
        ],
    },
    zip_safe=False,
)
