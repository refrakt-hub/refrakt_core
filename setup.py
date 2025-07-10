from setuptools import find_packages, setup

setup(
    name="refrakt",
    version="0.3",
    packages=find_packages(where="src"),
    include_package_data=True,
    package_dir={
        "refrakt": ""
    },  # This maps the root level modules to refrakt namespace
)
