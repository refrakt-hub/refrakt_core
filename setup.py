from setuptools import find_packages, setup

setup(
    name="refrakt",
    version="0.1",
    packages=find_packages(),
    package_dir={
        "refrakt": ""
    },  # This maps the root level modules to refrakt namespace
)
