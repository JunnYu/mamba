# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import os
from site import getsitepackages
from paddle.utils.cpp_extension import CUDAExtension, setup

this_dir = os.path.dirname(os.path.abspath(__file__))


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


paddle_includes = [
    os.path.dirname(os.path.abspath(__file__)),
    str(Path(this_dir) / "csrc" / "selective_scan"),
]
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include"))
    paddle_includes.append(
        os.path.join(site_packages_path, "paddle", "include", "third_party")
    )

sources = [
    "csrc/selective_scan/selective_scan.cpp",
    # fp32
    "csrc/selective_scan/selective_scan_fwd_fp32.cu",
    "csrc/selective_scan/selective_scan_bwd_fp32.cu",
    # fp16
    "csrc/selective_scan/selective_scan_fwd_fp16.cu",
    "csrc/selective_scan/selective_scan_bwd_fp16.cu",
    # bf16
    "csrc/selective_scan/selective_scan_fwd_bf16.cu",
    "csrc/selective_scan/selective_scan_bwd_bf16.cu",
]

arch_list = ["80"]
cc_flag = []
for arch in arch_list:
    cc_flag.append("-gencode")
    cc_flag.append(f"arch=compute_{arch},code=sm_{arch}")


extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": append_nvcc_threads(
        [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
        ]
        + cc_flag
    ),
}

setup(
    name="selective_scan_cuda_paddle",
    ext_modules=CUDAExtension(
        sources=sources,
        include_dirs=paddle_includes,
        extra_compile_args=extra_compile_args,
        verbose=True,
    ),
)
