"""
Setup file for Clinical Trial Qualification ML Model
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clinical-trial-qualification",
    version="1.0.0",
    author="Pragya J",
    author_email="pragya@example.com",
    description="ML model for clinical trial patient eligibility classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PragyaJ23/clinical-trial-qualification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)
