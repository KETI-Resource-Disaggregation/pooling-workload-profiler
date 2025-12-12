# Pooling Workload Profiler - Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_86 -cudart shared
PYTHON = python3

.PHONY: all clean test run-api

all: profiler

# ==================== Kernel Profiler ====================
profiler:
	@echo "Building kernel profiler..."
	$(NVCC) $(NVCC_FLAGS) -o profiler/kernel_profiler profiler/kernel_profiler.cu

# ==================== Clean ====================
clean:
	@echo "Cleaning..."
	rm -f profiler/kernel_profiler

# ==================== Test ====================
test: all
	@echo "Running kernel profiler..."
	cd profiler && ./kernel_profiler

# ==================== Run API Server ====================
run-api:
	@echo "Starting Profiler API Server..."
	$(PYTHON) api/profiler_api.py
