import os
import sys

from setuptools import Extension, setup

try:
    import pybind11
except Exception as exc:  # pragma: no cover - build-time guard
    raise SystemExit(
        "pybind11 is required to build _ogc2026_ckernel; "
        "use build_codespace.sh for the pinned build"
    ) from exc


debug = os.environ.get("OGC_CKERNEL_DEBUG") == "1"

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17"]
    extra_link_args = []
else:
    extra_compile_args = [
        "-std=c++17",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
    ]
    if debug:
        extra_compile_args += ["-O0", "-g", "-DOGC_CKERNEL_DEBUG"]
    else:
        extra_compile_args += ["-O3", "-DNDEBUG", "-fvisibility=hidden"]
    extra_link_args = []


setup(
    name="_ogc2026_ckernel",
    version="0.1.0",
    ext_modules=[
        Extension(
            "_ogc2026_ckernel",
            sources=["module.cpp", "feasible_map_core.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
