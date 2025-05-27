# CUDA Compilation Makefile
# Architecture: sm_90a with all optimizations enabled

# Variables
NVCC := $(shell if which pixi > /dev/null 2>&1; then echo "pixi run nvcc"; else echo "nvcc"; fi)
PYTHON := $(shell if which pixi > /dev/null 2>&1; then echo "pixi run python"; else echo "python"; fi)
ARCH := compute_90a
CODE := sm_90a
INCLUDES := -I. -I../3rdparty/cutlass/include
OPTIMIZATION_FLAGS := -O3

# Default target
.PHONY: default build benchmark submodule cubin so fmt clean

default: build benchmark

# Update submodule
submodule:
	git submodule update --init --recursive

# Compile kernel.cu into a cubin file
cubin: submodule
	$(NVCC) -gencode=arch=$(ARCH),code=$(CODE) $(OPTIMIZATION_FLAGS) $(INCLUDES) -cubin -o kernel.cubin kernel.cu

# Compile wrapped_kernel.cu into a shared library
so: submodule
	$(NVCC) -gencode=arch=$(ARCH),code=$(CODE) $(OPTIMIZATION_FLAGS) $(INCLUDES) -shared -Xcompiler -fPIC -o libwrapped_kernel.so wrapped_kernel.cu

# Compile both
build: cubin so

# Format code
fmt:
	$(PYTHON) -m isort .
	$(PYTHON) -m autopep8 --in-place --recursive .

# Run benchmark
benchmark:
	$(PYTHON) main.py

# Profile
profile:
	nsys profile -o nsys -f true -t nvtx,cuda,osrt $(PYTHON) main.py

# Clean targets
clean:
	rm -f kernel.cubin libwrapped_kernel.so 