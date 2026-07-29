// occ_calc — 이론 SM occupancy 계산기 (Exp_48 Phase 0).
//
// ★공식 구현 사용: NVIDIA CUDA Toolkit 의 cuda_occupancy.h
// (cudaOccMaxActiveBlocksPerMultiprocessor — CUDA Occupancy Calculator 의
//  참조 구현, 임의 근사 없음). 장치 속성은 cudaGetDeviceProperties 실값.
//
// usage: occ_calc <device_idx>
//   stdin  줄마다: <block_threads> <regs_per_thread> <smem_bytes>
//   stdout 줄마다: <max_active_blocks_per_sm> <active_warps> <occupancy>
//   첫 줄에 장치 헤더: DEV|<name>|sm_count|max_warps_per_sm|warp_size
//
// build: gcc -O2 occ_calc.c -o occ_calc -I/usr/local/cuda/include \
//        -L/usr/local/cuda/lib64 -lcudart
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

int main(int argc, char **argv) {
    int dev = (argc > 1) ? atoi(argv[1]) : 0;
    struct cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties 실패 (dev %d)\n", dev);
        return 1;
    }
    cudaOccDeviceProp occProp;
    memset(&occProp, 0, sizeof(occProp));
    occProp.computeMajor = p.major;
    occProp.computeMinor = p.minor;
    occProp.maxThreadsPerBlock = p.maxThreadsPerBlock;
    occProp.maxThreadsPerMultiprocessor = p.maxThreadsPerMultiProcessor;
    occProp.regsPerBlock = p.regsPerBlock;
    occProp.regsPerMultiprocessor = p.regsPerMultiprocessor;
    occProp.warpSize = p.warpSize;
    occProp.sharedMemPerBlock = p.sharedMemPerBlock;
    occProp.sharedMemPerMultiprocessor = p.sharedMemPerMultiprocessor;
    occProp.sharedMemPerBlockOptin = p.sharedMemPerBlockOptin;
    occProp.numSms = p.multiProcessorCount;
    occProp.reservedSharedMemPerBlock = p.reservedSharedMemPerBlock;

    cudaOccDeviceState state;
    memset(&state, 0, sizeof(state));
    // ★carveout=MAX_SHARED: 런타임은 커널의 smem 요구에 맞춰 carveout 을 자동
    // 상향하므로, 이론 occupancy 는 "필요 smem 이 확보된다" 가정이 물리적으로
    // 맞다 (0-init 은 MAX_L1=shared 최소로 해석되어 과소평가 — Exp_48 실측 확인).
    state.carveoutConfig = SHAREDMEM_CARVEOUT_MAX_SHARED;
    state.cacheConfig = CACHE_PREFER_SHARED;

    int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    printf("DEV|%s|%d|%d|%d\n", p.name, p.multiProcessorCount, max_warps,
           p.warpSize);

    int block, regs;
    long smem;
    while (scanf("%d %d %ld", &block, &regs, &smem) == 3) {
        cudaOccFuncAttributes attr;
        memset(&attr, 0, sizeof(attr));
        attr.maxThreadsPerBlock = p.maxThreadsPerBlock;
        attr.numRegs = regs;
        attr.sharedSizeBytes = 0;         // 트레이스는 static+dynamic 합산 →
        attr.partitionedGCConfig = PARTITIONED_GC_OFF;
        attr.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;   // flash-attn 류 대형 smem 허용
        attr.maxDynamicSharedSizeBytes = p.sharedMemPerBlockOptin;
        attr.numBlockBarriers = 1;
        cudaOccResult res;
        // 합산 smem 을 dynamic 인자로 전달 (총량만이 occupancy 에 유효)
        cudaOccError e = cudaOccMaxActiveBlocksPerMultiprocessor(
            &res, &occProp, &attr, &state, block, (size_t)smem);
        if (e != CUDA_OCC_SUCCESS) {
            printf("0 0 0.0\n");
            continue;
        }
        int blocks = res.activeBlocksPerMultiprocessor;
        int warps_per_block = (block + p.warpSize - 1) / p.warpSize;
        int active_warps = blocks * warps_per_block;
        printf("%d %d %.4f\n", blocks, active_warps,
               (double)active_warps / (double)max_warps);
    }
    return 0;
}
