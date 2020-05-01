import os

from setuptools import setup, find_packages

import versioneer


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
    with open(file, encoding="utf8") as f:
        content = f.read()
    return content


requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements", 'install.txt'))
requirements_async = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements", 'install_async.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))

setup(
    name='rising',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url='https://github.com/phoenixdl/rising',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    extras_require={'async': requirements_async},
    tests_require=["coverage"],
    python_requires=">=3.6",
    author="PhoenixDL",
    maintainer='Michael Baumgartner, Justus Schock',
    maintainer_email='justus.schock@rwth-aachen.de',
    license='MIT',
)
