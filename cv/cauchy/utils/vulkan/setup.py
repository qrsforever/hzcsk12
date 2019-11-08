import re
import os
import io

from setuptools import find_packages, setup


def read(*names, **kwargs):
  with io.open(
      os.path.join(os.path.dirname(__file__), *names),
      encoding=kwargs.get("encoding", "utf8")) as fp:
    return fp.read()


def find_version(*file_paths):
  version_file = read(*file_paths)
  version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file,
                            re.M)
  if version_match:
    return version_match.group(1)
  raise RuntimeError("Unable to find version string.")


readme = read('README.md')

VERSION = find_version('vulkan', '__init__.py')

setup(
    name="vulkan",
    version=VERSION,
    author="SIGAI Vulkan Team",
    description="Compute Engine for Cauchy",
    long_description=readme,
    license='LICENSE',

    # pacakge info
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["torch", "torchvision", "protobuf"],
)
