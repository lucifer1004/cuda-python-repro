import ctypes
import subprocess
import sys

import cuda.bindings
import cuda.bindings.driver as cuda_driver
import torch

import kernel


def load_cubin_and_get_kernel(cubin_path, kernel_name):
    """Load CUBIN file and get kernel function using CUDA driver API"""
    # Initialize CUDA driver
    cuda_driver.cuInit(0)

    # Get device and create context
    device = cuda_driver.CUdevice(0)
    context = cuda_driver.cuCtxCreate(0, device)[1]

    # Load CUBIN file
    with open(cubin_path, "rb") as f:
        cubin_data = f.read()

    # Load module from CUBIN data
    library = cuda_driver.cuLibraryLoadFromFile(
        bytes(cubin_path, "utf-8"), [], [], 0, [], [], 0)[1]

    # Get kernel function
    kernel_func = cuda_driver.cuLibraryGetKernel(
        library, kernel_name.encode())[1]

    return {kernel_name: kernel_func}, context


def load_shared_library(so_path):
    """Load shared library using ctypes"""
    return ctypes.CDLL(so_path)


def create_test_tensors():
    """Create test tensors for benchmarking"""
    # Create tensors with the expected dimensions (512x512) and half precision
    A = torch.randn(512, 512, dtype=torch.float16, device='cuda')
    B = torch.randn(512, 512, dtype=torch.float16, device='cuda')
    C = torch.zeros(512, 512, dtype=torch.float16, device='cuda')
    return A, B, C


def benchmark_cubin_approach(num_iterations=100, warmup_iterations=10):
    """Benchmark the kernel.py + kernel.cubin approach"""
    print("Benchmarking CUBIN approach (kernel.py + kernel.cubin)...")

    # Load CUBIN and get kernel
    kernels, context = load_cubin_and_get_kernel("kernel.cubin", "main_kernel")

    # Create test tensors
    A, B, C = create_test_tensors()


    # Warmup runs
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        kernel.call(kernels, A, B, C)

    # Benchmark runs with separate event pairs for each iteration
    print(f"Running {num_iterations} benchmark iterations...")
    torch.cuda.synchronize()

    iteration_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for i in range(num_iterations):
        # Record start event
        start_event.record()
        
        # Execute kernel
        kernel.call(kernels, A, B, C)
        
        # Record end event
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate elapsed time for this iteration
        elapsed_ms = start_event.elapsed_time(end_event)
        iteration_times.append(elapsed_ms / 1000.0)  # Convert to seconds

    # Calculate statistics
    total_time = sum(iteration_times)
    avg_time = total_time / num_iterations

    return total_time, avg_time


def benchmark_shared_library_approach(num_iterations=100, warmup_iterations=10):
    """Benchmark the libwrapped_kernel.so approach"""
    print("Benchmarking shared library approach (libwrapped_kernel.so)...")

    # Load shared library
    lib = load_shared_library("./libwrapped_kernel.so")

    # Set up function signatures
    lib.init.restype = ctypes.c_int
    lib.call.restype = ctypes.c_int
    lib.call.argtypes = [ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.get_last_error.restype = ctypes.c_char_p

    # Initialize the library
    init_result = lib.init()
    if init_result != 0:
        error_msg = lib.get_last_error().decode('utf-8')
        raise RuntimeError(f"Failed to initialize shared library: {error_msg}")

    # Create test tensors
    A, B, C = create_test_tensors()

    # Create CUDA stream
    stream = torch.cuda.current_stream().cuda_stream

    # Get tensor data pointers
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()

    # Warmup runs
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        result = lib.call(A_ptr, B_ptr, C_ptr, stream)
        if result != 0:
            error_msg = lib.get_last_error().decode('utf-8')
            raise RuntimeError(f"Kernel call failed: {error_msg}")
        torch.cuda.synchronize()

    # Benchmark runs with separate event pairs for each iteration
    print(f"Running {num_iterations} benchmark iterations...")
    torch.cuda.synchronize()

    iteration_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for i in range(num_iterations):
        # Record start event
        start_event.record()
        
        # Execute kernel
        result = lib.call(A_ptr, B_ptr, C_ptr, stream)
        if result != 0:
            error_msg = lib.get_last_error().decode('utf-8')
            raise RuntimeError(f"Kernel call failed: {error_msg}")
        
        # Record end event
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate elapsed time for this iteration
        elapsed_ms = start_event.elapsed_time(end_event)
        iteration_times.append(elapsed_ms / 1000.0)  # Convert to seconds

    # Calculate statistics
    total_time = sum(iteration_times)
    avg_time = total_time / num_iterations

    return total_time, avg_time


def benchmark():
    """Main benchmark function comparing both approaches"""
    print("CUDA Kernel Benchmark: Comparing CUBIN vs Shared Library Approaches")
    print("=" * 70)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    print(f"Using CUDA device: {torch.cuda.get_device_name()}")

    # Python version
    python_version = sys.version
    print(f"Python version: {python_version}")

    # NVCC Version
    nvcc_version = subprocess.check_output(
        ["nvcc", "--version"], text=True).split("\n")[-2].split()[1]
    print(f"NVCC version: {nvcc_version}")

    # CUDA Python version
    cuda_python_version = cuda.bindings.__version__
    print(f"CUDA Python version: {cuda_python_version}")

    num_iterations = 100
    warmup_iterations = 10

    try:
        # Benchmark CUBIN approach
        cubin_total, cubin_avg = benchmark_cubin_approach(
            num_iterations, warmup_iterations)
        print(
            f"CUBIN approach - Total time: {cubin_total:.6f}s, Average time: {cubin_avg*1000:.3f}ms")
        print()

        # Benchmark shared library approach
        so_total, so_avg = benchmark_shared_library_approach(
            num_iterations, warmup_iterations)
        print(
            f"Shared library approach - Total time: {so_total:.6f}s, Average time: {so_avg*1000:.3f}ms")
        print()

        # Compare results
        print("Performance Comparison:")
        print("-" * 30)
        if cubin_avg < so_avg:
            speedup = so_avg / cubin_avg
            print(f"CUBIN approach is {speedup:.2f}x faster")
        else:
            speedup = cubin_avg / so_avg
            print(f"Shared library approach is {speedup:.2f}x faster")

        print(
            f"Time difference: {abs(cubin_avg - so_avg)*1000:.3f}ms per call")

    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    benchmark()
