/**
 * Kernel Profiler - Compute/Memory Bound Classification
 *
 * Based on Orion research: Co-locate compute-bound + memory-bound kernels
 * for better GPU utilization.
 *
 * Key metrics for classification:
 * - IPC (Instructions Per Cycle) - high = compute-bound
 * - Memory throughput ratio - high = memory-bound
 * - Arithmetic intensity (FLOP/byte) - high = compute-bound
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

// ============================================================
// Kernel Type Definitions
// ============================================================
typedef enum {
    KERNEL_TYPE_UNKNOWN = 0,
    KERNEL_TYPE_COMPUTE_BOUND = 1,
    KERNEL_TYPE_MEMORY_BOUND = 2,
    KERNEL_TYPE_MIXED = 3
} KernelType;

typedef struct {
    const char* name;
    KernelType type;
    float duration_us;          // Kernel duration in microseconds
    float achieved_occupancy;   // SM occupancy (0-1)
    float memory_throughput_gb; // Memory throughput in GB/s
    float compute_throughput_gflops; // Compute throughput in GFLOPS
    float arithmetic_intensity; // FLOP per byte
    float ipc_estimate;         // Estimated IPC
} KernelProfile;

// ============================================================
// Test Kernels - Compute Bound
// ============================================================

// Heavy FMA (Fused Multiply-Add) kernel - pure compute
__global__ void kernel_compute_heavy(float* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = data[idx];
    float a = 1.00001f;
    float b = 0.99999f;

    // Heavy computation with no memory access
    #pragma unroll 16
    for (int i = 0; i < iterations; i++) {
        val = fmaf(val, a, b);  // FMA instruction
        val = fmaf(val, b, a);
        val = fmaf(val, a, b);
        val = fmaf(val, b, a);
    }

    data[idx] = val;
}

// Matrix multiply tile - compute intensive
__global__ void kernel_matmul_compute(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + 31) / 32; t++) {
        if (row < M && t * 32 + tx < K)
            As[ty][tx] = A[row * K + t * 32 + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && t * 32 + ty < K)
            Bs[ty][tx] = B[(t * 32 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 32; k++) {
            sum = fmaf(As[ty][k], Bs[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Transcendental functions - very compute heavy
__global__ void kernel_transcendental(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = data[idx];

    // Heavy transcendental operations
    #pragma unroll 8
    for (int i = 0; i < 50; i++) {
        val = sinf(val) + cosf(val);
        val = expf(-fabsf(val) * 0.01f);
        val = sqrtf(fabsf(val) + 1.0f);
    }

    data[idx] = val;
}

// ============================================================
// Test Kernels - Memory Bound
// ============================================================

// Simple copy kernel - pure memory bandwidth
__global__ void kernel_memory_copy(float* dst, const float* src, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    dst[idx] = src[idx];
}

// Strided memory access - memory latency bound
__global__ void kernel_memory_strided(float* data, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_idx = (idx * stride) % N;
    if (actual_idx < N) {
        data[actual_idx] = data[actual_idx] + 1.0f;
    }
}

// Random access pattern - very memory latency bound
__global__ void kernel_memory_random(float* data, const int* indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int target = indices[idx];
    data[target] = data[target] + 1.0f;
}

// Reduction kernel - mixed but memory dominant
__global__ void kernel_reduction(float* input, float* output, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// Test Kernels - Mixed Workloads
// ============================================================

// Stencil operation - balanced compute and memory
__global__ void kernel_stencil_2d(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        float val = input[idx - width] + input[idx + width] +
                    input[idx - 1] + input[idx + 1] +
                    4.0f * input[idx];
        output[idx] = val * 0.125f;
    }
}

// Histogram - mixed access pattern
__global__ void kernel_histogram(int* histogram, const unsigned char* data, int N, int bins) {
    extern __shared__ int local_hist[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize local histogram
    for (int i = tid; i < bins; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Count
    if (idx < N) {
        int bin = data[idx] % bins;
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    // Write back
    for (int i = tid; i < bins; i += blockDim.x) {
        atomicAdd(&histogram[i], local_hist[i]);
    }
}

// ============================================================
// Profiler Utilities
// ============================================================

inline int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

class KernelProfiler {
private:
    cudaEvent_t start_event, stop_event;
    std::vector<KernelProfile> profiles;

public:
    KernelProfiler() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~KernelProfiler() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void startTiming() {
        cudaEventRecord(start_event);
    }

    float stopTiming() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms * 1000.0f; // Return microseconds
    }

    // Profile a compute-heavy kernel
    KernelProfile profileComputeKernel(int N, int iterations) {
        float* d_data;
        cudaMalloc(&d_data, N * sizeof(float));

        // Warmup
        kernel_compute_heavy<<<(N + 255) / 256, 256>>>(d_data, N, iterations);
        cudaDeviceSynchronize();

        // Profile
        startTiming();
        kernel_compute_heavy<<<(N + 255) / 256, 256>>>(d_data, N, iterations);
        float duration_us = stopTiming();

        // Calculate metrics
        // Each thread does iterations * 4 FMAs = iterations * 8 FLOPS
        int64_t total_flops = (int64_t)N * iterations * 8;
        float gflops = (float)total_flops / (duration_us * 1000.0f);  // GFLOPS

        // Memory: just initial read and final write
        int64_t total_bytes = (int64_t)N * sizeof(float) * 2;
        float gb_s = (float)total_bytes / (duration_us * 1000.0f);  // GB/s

        float arithmetic_intensity = (float)total_flops / (float)total_bytes;

        KernelProfile profile = {
            .name = "compute_heavy",
            .type = KERNEL_TYPE_COMPUTE_BOUND,
            .duration_us = duration_us,
            .achieved_occupancy = 0.0f,  // Would need CUPTI for accurate value
            .memory_throughput_gb = gb_s,
            .compute_throughput_gflops = gflops,
            .arithmetic_intensity = arithmetic_intensity,
            .ipc_estimate = gflops > 100.0f ? 2.0f : 1.0f  // Heuristic
        };

        cudaFree(d_data);
        profiles.push_back(profile);
        return profile;
    }

    // Profile memory copy kernel
    KernelProfile profileMemoryCopyKernel(int N) {
        float *d_src, *d_dst;
        cudaMalloc(&d_src, N * sizeof(float));
        cudaMalloc(&d_dst, N * sizeof(float));

        // Warmup
        kernel_memory_copy<<<(N + 255) / 256, 256>>>(d_dst, d_src, N);
        cudaDeviceSynchronize();

        // Profile
        startTiming();
        kernel_memory_copy<<<(N + 255) / 256, 256>>>(d_dst, d_src, N);
        float duration_us = stopTiming();

        // Calculate metrics
        int64_t total_bytes = (int64_t)N * sizeof(float) * 2;  // Read + Write
        float gb_s = (float)total_bytes / (duration_us * 1000.0f);

        // Very low compute
        float gflops = 0.0f;
        float arithmetic_intensity = 0.0f;  // No compute per byte

        KernelProfile profile = {
            .name = "memory_copy",
            .type = KERNEL_TYPE_MEMORY_BOUND,
            .duration_us = duration_us,
            .achieved_occupancy = 0.0f,
            .memory_throughput_gb = gb_s,
            .compute_throughput_gflops = gflops,
            .arithmetic_intensity = arithmetic_intensity,
            .ipc_estimate = 0.3f  // Low IPC for memory-bound
        };

        cudaFree(d_src);
        cudaFree(d_dst);
        profiles.push_back(profile);
        return profile;
    }

    // Profile strided memory access
    KernelProfile profileStridedMemoryKernel(int N, int stride) {
        float* d_data;
        cudaMalloc(&d_data, N * sizeof(float));
        cudaMemset(d_data, 0, N * sizeof(float));

        // Warmup
        kernel_memory_strided<<<(N + 255) / 256, 256>>>(d_data, N, stride);
        cudaDeviceSynchronize();

        // Profile
        startTiming();
        kernel_memory_strided<<<(N + 255) / 256, 256>>>(d_data, N, stride);
        float duration_us = stopTiming();

        int64_t total_bytes = (int64_t)N * sizeof(float) * 2;
        float gb_s = (float)total_bytes / (duration_us * 1000.0f);

        KernelProfile profile = {
            .name = "memory_strided",
            .type = KERNEL_TYPE_MEMORY_BOUND,
            .duration_us = duration_us,
            .achieved_occupancy = 0.0f,
            .memory_throughput_gb = gb_s,
            .compute_throughput_gflops = 0.0f,
            .arithmetic_intensity = 0.0f,
            .ipc_estimate = 0.2f
        };

        cudaFree(d_data);
        profiles.push_back(profile);
        return profile;
    }

    // Profile transcendental kernel
    KernelProfile profileTranscendentalKernel(int N) {
        float* d_data;
        cudaMalloc(&d_data, N * sizeof(float));

        // Initialize with some values
        float* h_data = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_data[i] = (float)(i % 100) * 0.01f;
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        free(h_data);

        // Warmup
        kernel_transcendental<<<(N + 255) / 256, 256>>>(d_data, N);
        cudaDeviceSynchronize();

        // Profile
        startTiming();
        kernel_transcendental<<<(N + 255) / 256, 256>>>(d_data, N);
        float duration_us = stopTiming();

        // sin, cos, exp, sqrt ~ 8 ops each, 50 iterations, 3 ops per iteration
        int64_t total_ops = (int64_t)N * 50 * 3 * 8;
        float gflops = (float)total_ops / (duration_us * 1000.0f);

        int64_t total_bytes = (int64_t)N * sizeof(float) * 2;
        float gb_s = (float)total_bytes / (duration_us * 1000.0f);

        KernelProfile profile = {
            .name = "transcendental",
            .type = KERNEL_TYPE_COMPUTE_BOUND,
            .duration_us = duration_us,
            .achieved_occupancy = 0.0f,
            .memory_throughput_gb = gb_s,
            .compute_throughput_gflops = gflops,
            .arithmetic_intensity = (float)total_ops / (float)total_bytes,
            .ipc_estimate = 1.5f  // Transcendental ops have lower IPC than FMA
        };

        cudaFree(d_data);
        profiles.push_back(profile);
        return profile;
    }

    // Profile stencil kernel
    KernelProfile profileStencilKernel(int width, int height) {
        int N = width * height;
        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));
        cudaMemset(d_input, 0, N * sizeof(float));

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        // Warmup
        kernel_stencil_2d<<<grid, block>>>(d_output, d_input, width, height);
        cudaDeviceSynchronize();

        // Profile
        startTiming();
        kernel_stencil_2d<<<grid, block>>>(d_output, d_input, width, height);
        float duration_us = stopTiming();

        // 5 reads + 1 write per element, 5 adds + 1 mul
        int64_t total_bytes = (int64_t)N * sizeof(float) * 6;
        int64_t total_ops = (int64_t)N * 6;

        float gb_s = (float)total_bytes / (duration_us * 1000.0f);
        float gflops = (float)total_ops / (duration_us * 1000.0f);

        KernelProfile profile = {
            .name = "stencil_2d",
            .type = KERNEL_TYPE_MIXED,
            .duration_us = duration_us,
            .achieved_occupancy = 0.0f,
            .memory_throughput_gb = gb_s,
            .compute_throughput_gflops = gflops,
            .arithmetic_intensity = (float)total_ops / (float)total_bytes,
            .ipc_estimate = 0.8f
        };

        cudaFree(d_input);
        cudaFree(d_output);
        profiles.push_back(profile);
        return profile;
    }

    // Profile reduction kernel
    KernelProfile profileReductionKernel(int N) {
        float *d_input, *d_output;
        int blocks = (N + 255) / 256;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, blocks * sizeof(float));

        // Warmup
        kernel_reduction<<<blocks, 256, 256 * sizeof(float)>>>(d_input, d_output, N);
        cudaDeviceSynchronize();

        // Profile
        startTiming();
        kernel_reduction<<<blocks, 256, 256 * sizeof(float)>>>(d_input, d_output, N);
        float duration_us = stopTiming();

        int64_t total_bytes = (int64_t)N * sizeof(float) + blocks * sizeof(float);
        int64_t total_ops = (int64_t)N;  // One add per element

        float gb_s = (float)total_bytes / (duration_us * 1000.0f);
        float gflops = (float)total_ops / (duration_us * 1000.0f);

        KernelProfile profile = {
            .name = "reduction",
            .type = KERNEL_TYPE_MEMORY_BOUND,  // Reductions are typically memory-bound
            .duration_us = duration_us,
            .achieved_occupancy = 0.0f,
            .memory_throughput_gb = gb_s,
            .compute_throughput_gflops = gflops,
            .arithmetic_intensity = (float)total_ops / (float)total_bytes,
            .ipc_estimate = 0.4f
        };

        cudaFree(d_input);
        cudaFree(d_output);
        profiles.push_back(profile);
        return profile;
    }

    // Classify kernel type based on arithmetic intensity
    static KernelType classifyKernel(float arithmetic_intensity, float memory_throughput_gb) {
        // Roofline model thresholds (rough estimates for modern GPUs)
        // Memory bound: AI < 10 FLOP/byte
        // Compute bound: AI > 50 FLOP/byte
        // Mixed: 10 <= AI <= 50

        if (arithmetic_intensity < 10.0f) {
            return KERNEL_TYPE_MEMORY_BOUND;
        } else if (arithmetic_intensity > 50.0f) {
            return KERNEL_TYPE_COMPUTE_BOUND;
        } else {
            return KERNEL_TYPE_MIXED;
        }
    }

    void printProfile(const KernelProfile& p) {
        const char* type_str;
        switch (p.type) {
            case KERNEL_TYPE_COMPUTE_BOUND: type_str = "COMPUTE"; break;
            case KERNEL_TYPE_MEMORY_BOUND: type_str = "MEMORY"; break;
            case KERNEL_TYPE_MIXED: type_str = "MIXED"; break;
            default: type_str = "UNKNOWN";
        }

        printf("%-20s | Type: %-8s | Time: %8.2f us | Mem: %8.2f GB/s | Compute: %8.2f GFLOPS | AI: %8.2f FLOP/B\n",
               p.name, type_str, p.duration_us, p.memory_throughput_gb,
               p.compute_throughput_gflops, p.arithmetic_intensity);
    }

    void printAllProfiles() {
        printf("\n========== Kernel Profile Summary ==========\n");
        printf("%-20s | %-8s | %-12s | %-12s | %-16s | %-12s\n",
               "Kernel", "Type", "Time", "Mem BW", "Compute", "Arith Intens");
        printf("---------------------------------------------------------------------------------------------------------------\n");
        for (const auto& p : profiles) {
            printProfile(p);
        }
        printf("=============================================\n\n");
    }

    const std::vector<KernelProfile>& getProfiles() const {
        return profiles;
    }
};

// ============================================================
// Co-location Recommender
// ============================================================

class ColocationRecommender {
public:
    // Check if two kernel types are good for co-location
    static bool shouldColocate(KernelType type1, KernelType type2) {
        // Based on Orion: Compute + Memory bound kernels benefit from co-location
        if ((type1 == KERNEL_TYPE_COMPUTE_BOUND && type2 == KERNEL_TYPE_MEMORY_BOUND) ||
            (type1 == KERNEL_TYPE_MEMORY_BOUND && type2 == KERNEL_TYPE_COMPUTE_BOUND)) {
            return true;
        }

        // Mixed kernels can co-locate with anything
        if (type1 == KERNEL_TYPE_MIXED || type2 == KERNEL_TYPE_MIXED) {
            return true;
        }

        // Same type kernels compete for same resources
        return false;
    }

    // Score potential co-location benefit (0-1, higher is better)
    static float colocationScore(const KernelProfile& k1, const KernelProfile& k2) {
        // Ideal: one uses compute, other uses memory
        float compute_balance = fabsf(k1.arithmetic_intensity - k2.arithmetic_intensity);
        float mem_balance = fabsf(k1.memory_throughput_gb - k2.memory_throughput_gb);

        // Normalize and combine
        float ai_score = fminf(compute_balance / 100.0f, 1.0f);

        // If both are similar, score is low
        if (k1.type == k2.type && k1.type != KERNEL_TYPE_MIXED) {
            return 0.2f;  // Low score for same-type
        }

        // Compute + Memory is ideal
        if ((k1.type == KERNEL_TYPE_COMPUTE_BOUND && k2.type == KERNEL_TYPE_MEMORY_BOUND) ||
            (k1.type == KERNEL_TYPE_MEMORY_BOUND && k2.type == KERNEL_TYPE_COMPUTE_BOUND)) {
            return 0.8f + ai_score * 0.2f;
        }

        return 0.5f + ai_score * 0.3f;
    }

    static void printRecommendation(const KernelProfile& k1, const KernelProfile& k2) {
        float score = colocationScore(k1, k2);
        bool recommend = shouldColocate(k1.type, k2.type);

        printf("Co-location: %s + %s\n", k1.name, k2.name);
        printf("  Score: %.2f | Recommendation: %s\n",
               score, recommend ? "GOOD" : "AVOID");
        if (recommend) {
            printf("  Reason: Complementary resource usage (compute vs memory)\n");
        } else {
            printf("  Reason: Competing for same resources\n");
        }
        printf("\n");
    }
};

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    printf("=================================================\n");
    printf("     Kernel Profiler - Compute/Memory Bound     \n");
    printf("     Classification for Co-location Scheduling  \n");
    printf("=================================================\n\n");

    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d, Memory: %.2f GB, Bandwidth: %.2f GB/s\n",
           prop.multiProcessorCount,
           (float)prop.totalGlobalMem / (1024*1024*1024),
           (float)(prop.memoryClockRate * 1000 * (prop.memoryBusWidth / 8) * 2) / 1e9);
    printf("\n");

    KernelProfiler profiler;

    int N = 10 * 1024 * 1024;  // 10M elements
    int iterations = 100;

    printf("Profiling kernels with N = %d elements...\n\n", N);

    // Profile various kernel types
    printf("1. Compute-bound kernels:\n");
    auto p_compute = profiler.profileComputeKernel(N, iterations);
    profiler.printProfile(p_compute);

    auto p_trans = profiler.profileTranscendentalKernel(N);
    profiler.printProfile(p_trans);

    printf("\n2. Memory-bound kernels:\n");
    auto p_copy = profiler.profileMemoryCopyKernel(N);
    profiler.printProfile(p_copy);

    auto p_strided = profiler.profileStridedMemoryKernel(N, 16);
    profiler.printProfile(p_strided);

    auto p_reduction = profiler.profileReductionKernel(N);
    profiler.printProfile(p_reduction);

    printf("\n3. Mixed kernels:\n");
    auto p_stencil = profiler.profileStencilKernel(4096, 4096);
    profiler.printProfile(p_stencil);

    // Print summary
    profiler.printAllProfiles();

    // Co-location recommendations
    printf("========== Co-location Recommendations ==========\n\n");

    ColocationRecommender::printRecommendation(p_compute, p_copy);
    ColocationRecommender::printRecommendation(p_compute, p_trans);
    ColocationRecommender::printRecommendation(p_copy, p_reduction);
    ColocationRecommender::printRecommendation(p_stencil, p_compute);
    ColocationRecommender::printRecommendation(p_stencil, p_copy);

    printf("=================================================\n");
    printf("Summary:\n");
    printf("  - Compute-bound + Memory-bound: GOOD for co-location\n");
    printf("  - Same type kernels: AVOID co-location (resource contention)\n");
    printf("  - Mixed kernels: Can co-locate with either type\n");
    printf("=================================================\n");

    return 0;
}
