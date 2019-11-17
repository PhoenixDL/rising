from setuptools import setup, find_packages
import versioneer
import os
import re


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


requirements = resolve_requirements(os.path.join(os.path.dirname(__file__),
                                                 'requirements.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))
license = read_file(os.path.join(os.path.dirname(__file__), "LICENSE"))


setup(
    name='python_template',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    # url='path/to/url',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.7",
    author="Michael Baumgartner",
    author_email="michael.baumgartner@rwth-aachen.de",
    license=license,
)
