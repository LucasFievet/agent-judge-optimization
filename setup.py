"""Setup script for expforge training application.

This setup.py is required for creating Python source distributions
that can be used with Vertex AI prebuilt containers.
Dependencies are read from pyproject.toml to avoid duplication.
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="expforge",
    version="0.1.0",
    description="Lightweight experiment scaffolding for ML workflows",
    author="Experiment Forge",
    # Find all packages in the src directory
    # This will include: expforge, expforge.data, expforge.model, 
    # expforge.training, expforge.vertex, etc.
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    # Dependencies are read from pyproject.toml [project] section
    python_requires=">=3.10",
)

