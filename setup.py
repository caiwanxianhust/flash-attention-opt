from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob
os.environ['MAX_JOBS'] = '1'  # 限制为单进程编译

ROOT = os.path.dirname(os.path.abspath(__file__))

cu_files = glob.glob("flash_attention/*.cu")
binding_files = ["pytorch_binding/csrc/binding.cpp",]

setup(
    name="flash_attn",
    version="0.1.0",
    packages=["flash_attn"],
    package_dir={"flash_attn": "pytorch_binding/flash_attn"},
    ext_modules=[
        CUDAExtension(
            name="flash_attn._C",
            sources = cu_files + binding_files,
           
             
            # 【新增】在这里指定 CUTLASS 的头文件目录
            include_dirs=[
                f"{ROOT}/flash_attention",
                f"{ROOT}/pytorch_binding/csrc",
                "/usr/local/cutlass/include",
                "/usr/local/cutlass/tools/util/include",
                "/usr/local/cutlass/examples/common",  # 如果代码中用到了 helper.h 等
            ],
            
            extra_compile_args={
                "nvcc": [
                    # "-O0",  # 降低优化级别减少内存占用
                    "-arch=sm_89",  # 替换为你的GPU架构（如sm_61, sm_86）
                    "-std=c++17"
                ],
                "cxx": [
                    # "-O0", 
                    "-std=c++17"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

# python3 setup.py build_ext --inplace