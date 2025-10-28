# CUDA Python TMA Descriptor Creation Benchmark

This repository contains a **focused performance comparison of TMA (Tensor Memory Accelerator) descriptor creation** between CUDA Python bindings and native CUDA C++.

## 🎯 Purpose

**Measure and compare the overhead of `cuTensorMapEncodeTiled` API calls:**

- **Python**: CUDA Python bindings (`cuda.bindings.driver`)
- **C++**: Native CUDA Driver API

This benchmark isolates TMA descriptor creation to precisely quantify the Python binding overhead, without the noise of kernel launches or GPU execution.

## 🏗️ Repository Structure

```
cuda-python-repro/
├── issue/                      # Main benchmark code
│   ├── main.py                 # Benchmark script (Python & C++ comparison)
│   ├── wrapped_kernel.cu       # C++ TMA descriptor benchmark
│   ├── libwrapped_kernel.so    # Compiled shared library (generated)
│   ├── Makefile                # Build configuration (supports cu12/cu13)
│   └── Justfile                # Just build automation (supports cu12/cu13)
├── pixi.toml                   # Pixi package manager configuration
├── pixi.lock                   # Locked dependencies
└── README.md                   # This file
```

## 🔧 Prerequisites

- **CUDA Toolkit**: Version 12.x or 13.x
- **GPU**: NVIDIA GPU with compute capability 9.0a (H100/H200 series) 
- **Python**: 3.10 or higher
- **Pixi**: Package manager (recommended)

## 🚀 Quick Start

### Using Pixi (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cuda-python-repro
   ```

2. **Install dependencies**:
   ```bash
   pixi install
   ```

3. **Build and run (CUDA 12.x)**:
   ```bash
   cd issue
   make cu12
   # or: just cu12
   ```

4. **Build and run (CUDA 13.x)**:
   ```bash
   cd issue
   make cu13
   # or: just cu13
   ```

### Using Manual Setup

1. **Install dependencies**:
   ```bash
   pip install cuda-python torch
   ```

2. **Build and run**:
   ```bash
   cd issue
   nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -shared -Xcompiler -fPIC -lcuda -o libwrapped_kernel.so wrapped_kernel.cu
   python main.py
   ```

## 📊 Benchmark Details

The benchmark creates TMA descriptors for three 512×512 half-precision tensors:

- **Operation**: `cuTensorMapEncodeTiled` (3 descriptors per iteration)
- **Tensor Config**: 512×512 fp16, 64×64 tiles, swizzle 128B
- **Iterations**: 1000 iterations
- **Timing**: CPU-side timing with `time.perf_counter()` (Python) and `std::chrono` (C++)
- **Metrics**: Average time per iteration (microseconds)

### Expected Output

```
TMA Descriptor Creation Benchmark: Python vs C++
======================================================================
Using CUDA device: NVIDIA H100 80GB HBM3
Python version: 3.12.10
NVCC version: cuda_12.9.r12.9/compiler.35813241_0
CUDA Python version: 12.9.0

Benchmarking Python TMA descriptor creation (1000 iterations)...
Python - Total: 0.123456s, Average: 123.456μs per iteration

Benchmarking C++ TMA descriptor creation (1000 iterations)...
C++    - Total: 0.012345s, Average: 12.345μs per iteration

Performance Comparison:
------------------------------
C++ is 10.00x faster than Python
Time difference: 111.111μs per iteration
```

## 🛠️ Available Commands

### Using Make

```bash
make                    # Build and benchmark with cu12 (default)
make cu12               # Build and benchmark with CUDA 12.x
make cu13               # Build and benchmark with CUDA 13.x
make build              # Compile shared library only
make benchmark          # Run benchmark only
make clean              # Remove compiled artifacts
make fmt                # Format Python code
make profile            # Profile with nsys
make help               # Show all available targets

# Override environment
PIXI_ENV=cu13 make build
```

### Using Just

```bash
just                    # Build and benchmark with cu12 (default)
just cu12               # Build and benchmark with CUDA 12.x
just cu13               # Build and benchmark with CUDA 13.x
just build              # Compile shared library only
just benchmark          # Run benchmark only
just clean              # Remove compiled artifacts
just fmt                # Format Python code
just profile            # Profile with nsys

# Override environment
PIXI_ENV=cu13 just build
```

## 🔍 Key Components

### `main.py`
The main benchmark script that:
- Initializes CUDA context (compatible with cu12/cu13)
- Creates test tensors (512×512 fp16)
- Benchmarks Python TMA descriptor creation via `cuTensorMapEncodeTiled`
- Benchmarks C++ TMA descriptor creation via shared library
- Compares and reports performance metrics

### `wrapped_kernel.cu`
C++ benchmark function that:
- Implements `benchmark_tma_creation()` exported function
- Creates 3 TMA descriptors per iteration using CUDA Driver API
- Times the operations with `std::chrono`
- Returns average time per iteration

**Code size**: 97 lines (stripped down from 242 lines)

## 🐛 Troubleshooting

### Common Issues

1. **`undefined symbol: cuTensorMapEncodeTiled`**: 
   - Ensure `-lcuda` is in the compile command
   - Rebuild: `make clean && make build`

2. **CUDA version mismatch**: 
   - Use `make cu12` for CUDA 12.x or `make cu13` for CUDA 13.x
   - Check: `pixi run -e cu12 nvcc --version`

3. **Architecture mismatch**: 
   - Code targets `sm_90a` (H100/H200)
   - For other GPUs: modify `ARCH` and `CODE` in Makefile

4. **Import errors**: 
   - Verify: `pixi run -e cu12 python -c "import cuda.bindings; print(cuda.bindings.__version__)"`

## 📝 Dependencies

### CUDA 12.x Environment (cu12)
- **cuda-python**: 12.6.x - CUDA Python bindings
- **pytorch-gpu**: >=2.7.0,<3 - PyTorch with CUDA support
- **cuda-toolkit**: 12.6.x - NVIDIA CUDA Toolkit

### CUDA 13.x Environment (cu13)
- **cuda-python**: 13.0.x - CUDA Python bindings  
- **pytorch-gpu**: >=2.7.0,<3 - PyTorch with CUDA support
- **cuda-toolkit**: 13.0.x - NVIDIA CUDA Toolkit

All dependencies are managed via `pixi.toml`.

## 🤝 Contributing

This is a focused benchmark repository. If you encounter issues or have improvements:

1. Test with both `cu12` and `cu13` environments
2. Document any modifications for different GPU architectures
3. Report findings with system specifications and timing results

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🔗 Related Links

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [TMA (Tensor Memory Accelerator) Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator-tma)
