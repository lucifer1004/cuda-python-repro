# CUDA Compilation Makefile
# Architecture: sm_90a with all optimizations enabled
#
# Usage:
#   make           - Build with cu12 environment (default)
#   make cu12      - Build with cu12 environment
#   make cu13      - Build with cu13 environment
#   PIXI_ENV=cu13 make build - Override environment

# Variables
PIXI_ENV ?= cu12
NVCC := $(shell if which pixi > /dev/null 2>&1; then echo "pixi run -e $(PIXI_ENV) nvcc"; else echo "nvcc"; fi)
PYTHON := $(shell if which pixi > /dev/null 2>&1; then echo "pixi run -e $(PIXI_ENV) python"; else echo "python"; fi)
ARCH := compute_90a
CODE := sm_90a
OPTIMIZATION_FLAGS := -O3

# Default target
.PHONY: default build benchmark so fmt clean cu12 cu13 help

default: build benchmark

# Help target
help:
	@echo "Available targets:"
	@echo "  make / make cu12    - Build and benchmark with cu12 environment (default)"
	@echo "  make cu13           - Build and benchmark with cu13 environment"
	@echo "  make build          - Build shared library only"
	@echo "  make benchmark      - Run benchmark only"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make fmt            - Format Python code"
	@echo "  make profile        - Profile with nsys"
	@echo ""
	@echo "Environment: $(PIXI_ENV)"

# Environment-specific targets
cu12:
	$(MAKE) build benchmark PIXI_ENV=cu12

cu13:
	$(MAKE) build benchmark PIXI_ENV=cu13

# Compile wrapped_kernel.cu into a shared library
so:
	$(NVCC) -gencode=arch=$(ARCH),code=$(CODE) $(OPTIMIZATION_FLAGS) -shared -Xcompiler -fPIC -lcuda -o libwrapped_kernel.so wrapped_kernel.cu

# Build
build: so

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
	rm -f libwrapped_kernel.so