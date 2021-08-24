import os

from setuptools import find_packages, setup

import versioneer

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

builtins.__RISING_SETUP__ = True


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file, encoding="utf8") as f:
        content = f.read()
    return content


requirements = resolve_requirements(os.path.join(os.path.dirname(__file__), "requirements", "install.txt"))
requirements_async = resolve_requirements(os.path.join(os.path.dirname(__file__), "requirements", "install_async.txt"))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md")).replace(".svg", ".png")

import rising  # noqa: E402

setup(
    name="rising",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url=rising.__homepage__,
    test_suite="unittest",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={"async": requirements_async},
    tests_require=["coverage"],
    python_requires=">=3.6",
    author=rising.__author__,
    author_email=rising.__author_email__,
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "augmentation", "transforms", "pytorch", "medical"],
    license=rising.__license__,
    project_urls={
        "Bug Tracker": "https://github.com/PhoenixDL/rising/issues",
        "Documentation": "https://rising.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PhoenixDL/rising",
    },
    # TODO: Populate classifiers
    classifiers=[],
)
