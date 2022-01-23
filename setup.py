import os
import re

from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "ruatd", "__version__.py"),
    encoding="utf-8",
) as fp:
    VERSION = (
        re.compile(r""".*__VERSION__ = ["'](.*?)['"]""", re.S).match(fp.read()).group(1)
    )


setup(
    name="ruatd",
    version=VERSION,
    author="Narek Maloyan",
    author_email="narek1110@gmail.com",
    description="RuATD competition DIALOG-22",
    packages=find_packages(),
    package_dir={
        "ruatd": "ruatd",
    },
    install_requires=requirements,
    package_data={
        "": ["config/dev.yaml", "config/prod.yaml"],
    },
)
