import ctypes
import time
import subprocess
import sys

import cuda.bindings
from cuda.bindings.driver import (
    cuInit, CUdevice, cuCtxCreate,
    cuTensorMapEncodeTiled,
    CUtensorMapDataType, cuuint64_t, cuuint32_t,
    CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill,
    CUresult
)
import torch


def init_cuda():
    """Initialize CUDA driver"""
    cuInit(0)
    device = CUdevice(0)
    # Need to detect cuda version here and create context with the correct flags
    if cuda.bindings.__version__.split(".")[0] == "13":
        res, context = cuCtxCreate(None, 0, device)
    else:
        res, context = cuCtxCreate(0, device)
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to create CUDA context: {res}")
    return device, context


def load_shared_library(so_path):
    """Load shared library using ctypes"""
    return ctypes.CDLL(so_path)


def create_test_tensors():
    """Create test tensors"""
    A = torch.randn(512, 512, dtype=torch.float16, device='cuda')
    B = torch.randn(512, 512, dtype=torch.float16, device='cuda')
    C = torch.zeros(512, 512, dtype=torch.float16, device='cuda')
    return A, B, C


def benchmark_python_tma_creation(A, B, C, num_iterations=1000):
    """Benchmark TMA descriptor creation in Python"""
    print(f"Benchmarking Python TMA descriptor creation ({num_iterations} iterations)...")
    
    # Get tensor pointers once
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()
    
    # Pre-create constant parameters (avoid repeated construction overhead)
    data_type = CUtensorMapDataType(6)
    rank = 2
    global_dim = [cuuint64_t(512), cuuint64_t(512)]
    global_stride = [cuuint64_t(1024)]
    box_dim = [cuuint32_t(64), cuuint32_t(64)]
    element_strides = [cuuint32_t(1), cuuint32_t(1)]
    interleave = CUtensorMapInterleave(0)
    swizzle_3 = CUtensorMapSwizzle(3)
    swizzle_0 = CUtensorMapSwizzle(0)
    l2_promotion = CUtensorMapL2promotion(2)
    oob_fill = CUtensorMapFloatOOBfill(0)
    
    # Ensure GPU is idle before starting timing
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        # Create A descriptor
        res, A_desc = cuTensorMapEncodeTiled(
            data_type, rank, A_ptr, global_dim, global_stride,
            box_dim, element_strides, interleave, swizzle_3,
            l2_promotion, oob_fill
        )
        if res != CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create A_desc: {res}")
        
        # Create B descriptor
        res, B_desc = cuTensorMapEncodeTiled(
            data_type, rank, B_ptr, global_dim, global_stride,
            box_dim, element_strides, interleave, swizzle_3,
            l2_promotion, oob_fill
        )
        if res != CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create B_desc: {res}")
        
        # Create C descriptor
        res, C_desc = cuTensorMapEncodeTiled(
            data_type, rank, C_ptr, global_dim, global_stride,
            box_dim, element_strides, interleave, swizzle_0,
            l2_promotion, oob_fill
        )
        if res != CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create C_desc: {res}")
    
    end = time.perf_counter()
    total_time = end - start
    avg_time = total_time / num_iterations
    
    return total_time, avg_time


def benchmark_cpp_tma_creation(A, B, C, num_iterations=1000):
    """Benchmark TMA descriptor creation in C++ (measured from Python side)"""
    print(f"Benchmarking C++ TMA descriptor creation ({num_iterations} iterations)...")
    
    # Load shared library
    lib = load_shared_library("./libwrapped_kernel.so")
    
    # Set up function signatures for benchmark function
    lib.benchmark_tma_creation.restype = None
    lib.benchmark_tma_creation.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    
    # Get tensor data pointers
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()
    
    # Ensure GPU is idle before starting timing
    torch.cuda.synchronize()
    
    # Time the C++ function call from Python side (includes ctypes overhead)
    start = time.perf_counter()
    lib.benchmark_tma_creation(A_ptr, B_ptr, C_ptr, num_iterations)
    end = time.perf_counter()
    
    total_time = end - start
    avg_time = total_time / num_iterations
    
    return total_time, avg_time


def benchmark():
    """Main benchmark function comparing Python vs C++ TMA descriptor creation"""
    print("TMA Descriptor Creation Benchmark: Python vs C++")
    print("=" * 70)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"Python version: {sys.version.split()[0]}")
    
    try:
        nvcc_version = subprocess.check_output(
            ["nvcc", "--version"], text=True).split("\n")[-2].split()[1]
        print(f"NVCC version: {nvcc_version}")
    except:
        pass
    
    print(f"CUDA Python version: {cuda.bindings.__version__}")
    print()

    # Initialize CUDA
    device, context = init_cuda()
    
    # Create test tensors
    A, B, C = create_test_tensors()
    
    num_iterations = 1000

    try:
        # Benchmark Python approach
        py_total, py_avg = benchmark_python_tma_creation(A, B, C, num_iterations)
        print(f"Python - Total: {py_total:.6f}s, Average: {py_avg*1e6:.3f}μs per iteration")
        print()

        # Benchmark C++ approach
        cpp_total, cpp_avg = benchmark_cpp_tma_creation(A, B, C, num_iterations)
        print(f"C++    - Total: {cpp_total:.6f}s, Average: {cpp_avg*1e6:.3f}μs per iteration")
        print()

        # Compare results
        print("Performance Comparison:")
        print("-" * 30)
        speedup = py_avg / cpp_avg
        print(f"C++ is {speedup:.2f}x faster than Python")
        print(f"Time difference: {(py_avg - cpp_avg)*1e6:.3f}μs per iteration")

    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    benchmark()
