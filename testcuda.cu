#include "ising_gpu.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(-1);
    }
}

template<BoundaryCondition BC>
void runSharedVsGlobalUpdateTest(int size){
    int subSize = size / 2;
    int subLen = size * subSize;
    std::vector<int> h_whites(subLen);
    std::vector<int> h_blacks(subLen);
    for(int row = 0; row < size; row++){
        for(int col = 0; col < subSize; col++){
            int idx = row * subSize + col;
            // Use two different periodic integer patterns to build a mixed (non-uniform) spin state.
            h_whites[idx] = ((row + col) % 3 == 0) ? 1 : -1;
            h_blacks[idx] = ((2 * row + col) % 5 < 2) ? 1 : -1;
        }
    }

    int* d_whites_global;
    int* d_blacks_global;
    int* d_whites_shared;
    int* d_blacks_shared;
    checkCuda(cudaMalloc(&d_whites_global, subLen * sizeof(int)), "cudaMalloc d_whites_global failed");
    checkCuda(cudaMalloc(&d_blacks_global, subLen * sizeof(int)), "cudaMalloc d_blacks_global failed");
    checkCuda(cudaMalloc(&d_whites_shared, subLen * sizeof(int)), "cudaMalloc d_whites_shared failed");
    checkCuda(cudaMalloc(&d_blacks_shared, subLen * sizeof(int)), "cudaMalloc d_blacks_shared failed");
    checkCuda(cudaMemcpy(d_whites_global, h_whites.data(), subLen * sizeof(int), cudaMemcpyHostToDevice), "copy whites global failed");
    checkCuda(cudaMemcpy(d_blacks_global, h_blacks.data(), subLen * sizeof(int), cudaMemcpyHostToDevice), "copy blacks global failed");
    checkCuda(cudaMemcpy(d_whites_shared, h_whites.data(), subLen * sizeof(int), cudaMemcpyHostToDevice), "copy whites shared failed");
    checkCuda(cudaMemcpy(d_blacks_shared, h_blacks.data(), subLen * sizeof(int), cudaMemcpyHostToDevice), "copy blacks shared failed");

    curandStatePhilox4_32_10_t* d_state_global;
    curandStatePhilox4_32_10_t* d_state_shared;
    checkCuda(cudaMalloc(&d_state_global, subLen * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_state_global failed");
    checkCuda(cudaMalloc(&d_state_shared, subLen * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_state_shared failed");
    int threads1D = 256;
    int blocks1D = (subLen + threads1D - 1) / threads1D;
    // Use identical RNG initialization so global vs shared kernels can be compared bit-for-bit.
    initCurand<<<blocks1D, threads1D>>>(d_state_global, 42);
    initCurand<<<blocks1D, threads1D>>>(d_state_shared, 42);
    checkCuda(cudaGetLastError(), "initCurand launch failed");
    checkCuda(cudaDeviceSynchronize(), "initCurand execution failed");

    // Deterministic acceptance table: only deltaE == 0 flips are accepted.
    float expTable[5] = {0.f, 0.f, 1.f, 0.f, 0.f};
    checkCuda(cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float)), "copy exp table failed");

    updateSublatticeKernel<BC><<<blocks1D, threads1D>>>(d_whites_global, d_blacks_global, true, size, subSize, d_state_global);
    updateSublatticeKernel<BC><<<blocks1D, threads1D>>>(d_blacks_global, d_whites_global, false, size, subSize, d_state_global);

    dim3 threads2D(32, 16);
    dim3 blocks2D((subSize + threads2D.x - 1) / threads2D.x, (size + threads2D.y - 1) / threads2D.y);
    size_t shmemBytes = 2 * (threads2D.x + 2) * (threads2D.y + 2) * sizeof(int);
    int* d_mag = nullptr;
    checkCuda(cudaMalloc(&d_mag, sizeof(int)), "cudaMalloc d_mag failed");
    checkCuda(cudaMemset(d_mag, 0, sizeof(int)), "cudaMemset d_mag failed");
    updateSublatticeSharedKernel<BC><<<blocks2D, threads2D, shmemBytes>>>(d_whites_shared, d_blacks_shared, true, size, subSize, d_state_shared, d_mag);
    updateSublatticeSharedKernel<BC><<<blocks2D, threads2D, shmemBytes>>>(d_blacks_shared, d_whites_shared, false, size, subSize, d_state_shared, d_mag);
    checkCuda(cudaGetLastError(), "update kernels launch failed");
    checkCuda(cudaDeviceSynchronize(), "update kernels execution failed");

    std::vector<int> h_whites_global(subLen), h_blacks_global(subLen);
    std::vector<int> h_whites_shared(subLen), h_blacks_shared(subLen);
    checkCuda(cudaMemcpy(h_whites_global.data(), d_whites_global, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy whites global back failed");
    checkCuda(cudaMemcpy(h_blacks_global.data(), d_blacks_global, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy blacks global back failed");
    checkCuda(cudaMemcpy(h_whites_shared.data(), d_whites_shared, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy whites shared back failed");
    checkCuda(cudaMemcpy(h_blacks_shared.data(), d_blacks_shared, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy blacks shared back failed");

    for(int i = 0; i < subLen; i++){
        assert(h_whites_global[i] == h_whites_shared[i]);
        assert(h_blacks_global[i] == h_blacks_shared[i]);
    }

    int magShared = 0;
    checkCuda(cudaMemcpy(&magShared, d_mag, sizeof(int), cudaMemcpyDeviceToHost), "copy mag failed");
    int magHost = 0;
    for(int i = 0; i < subLen; i++){
        magHost += h_whites_shared[i] + h_blacks_shared[i];
    }
    assert(magShared == magHost);

    cudaFree(d_mag);
    cudaFree(d_state_global);
    cudaFree(d_state_shared);
    cudaFree(d_whites_global);
    cudaFree(d_blacks_global);
    cudaFree(d_whites_shared);
    cudaFree(d_blacks_shared);
}

int main(){
    runSharedVsGlobalUpdateTest<Open>(8);
    runSharedVsGlobalUpdateTest<Periodic>(8);
    std::cout << "Shared-memory update tests passed." << std::endl;
    return 0;
}
