import io
import os
import os.path as osp
import re
from glob import glob

import numpy as np
import torch
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "cauchy/models/rcnn", "csrc")

    main_file = glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "cauchy.models.rcnn._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


readme = "CV" # read("README.md")

VERSION = find_version("cauchy", "__init__.py")

install_requires = []

packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

setup(
    name="cauchy",
    version=VERSION,
    author="SIGAI Vulkan Team",
    description="CV backend for Cauchy",
    long_description=readme,
    license="LICENSE",
    # pacakge info
    packages=packages,
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    package_data={
        "cauchy.extensions.ops.nms": ["*.*.so"],
        "cauchy.extensions.ops.roi_align": ["*.*.so"],
        "cauchy.extensions.ops.roi_pool": ["*.*.so"],
        "cauchy.extensions.ops.dcn": ["*.*.so"],
        "cauchy.extensions.ops.inplace_abn": [
            "src/*.h",
            "src/*.cpp",
            "src/*.cu",
            "src/*.cuh",
            "src/utils/*.h",
            "src/utils/*.cuh",
        ],
        "cauchy.extensions.ops.syncbn": ["src/*.h", "src/*.cu", "src/.cpp"],
    },
)
