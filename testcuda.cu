#include "isingcuda.hpp"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << msg << ": " << cudaGetErrorString(err) << endl;
        exit(-1);
    }
}



int main(int argc, char** argv){

    if(argc != 2){
        cerr << "Usage: " << argv[0] << " <grid_size>" << endl;
        return -1;
    }
    int size = stoi(argv[1]);
    int subSize = size / 2;
    int subLen = size*subSize;

    //prepare whites up blacks down

    int* whites = new int[subLen];
    int* blacks = new int[subLen];
    for(int i = 0; i < subLen; i++){
        whites[i] = 1;
        blacks[i] = -1;
    }
    int* d_whites;
    int* d_blacks;
    checkCuda(cudaMalloc(&d_whites, subLen * sizeof(int)), "cudaMalloc d_whites failed");
    checkCuda(cudaMalloc(&d_blacks, subLen * sizeof(int)), "cudaMalloc d_blacks failed");
    checkCuda(cudaMemcpy(d_whites, whites, subLen * sizeof(int), cudaMemcpyHostToDevice), "copy whites to device failed");
    checkCuda(cudaMemcpy(d_blacks, blacks, subLen * sizeof(int), cudaMemcpyHostToDevice), "copy blacks to device failed");

    int nblocks = 1;
    int nthreads = 512;

    //initialize curand states
    curandState* d_state;
    checkCuda(cudaMalloc(&d_state, subLen * sizeof(curandState)), "cudaMalloc d_state failed");
    
    //low temp/botchered update to verify
    double temp = 0.01;
    
    //test with different block/thread size
    int nblocks_a[] = {1, 2, 4};
    
    for(int b{0}; b<3; b++){
        int requestedBlocks = nblocks_a[b];
        nblocks = requestedBlocks;
        nthreads = (subLen + nblocks - 1) / nblocks;
        if (nthreads > 1024) {
            cout << "Requested " << nthreads << " threads per block exceeds limit. Adjusting..." << endl;
            nthreads = 1024;
            nblocks = (subLen + nthreads - 1) / nthreads;
        }
        initCurand<<<nblocks, nthreads>>>(d_state, 42);
        checkCuda(cudaGetLastError(), "initCurand launch failed");
        checkCuda(cudaDeviceSynchronize(), "initCurand execution failed");
        cout << "testing with requested " << requestedBlocks << " blocks (actual " << nblocks << ") and " << nthreads << " threads" << endl;
        for(int i = 0; i < subLen; i++){
            whites[i] = 1;
            blacks[i] = -1;
        }
        checkCuda(cudaMemcpy(d_whites, whites, subLen * sizeof(int), cudaMemcpyHostToDevice), "reset whites copy failed");
        checkCuda(cudaMemcpy(d_blacks, blacks, subLen * sizeof(int), cudaMemcpyHostToDevice), "reset blacks copy failed");

        updateSublatticeKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, temp,  d_state);
        checkCuda(cudaGetLastError(), "update white launch failed");
        checkCuda(cudaDeviceSynchronize(), "update white execution failed");
        checkCuda(cudaMemcpy(whites, d_whites, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy whites to host failed");
        for(int i = 0; i < subLen; i++){
            if(whites[i] != -1){
                cerr << "found err white at index " << i << " " << whites[i] << endl;
                return -1;
            }
        }

        for(int i = 0; i < subLen; i++){
            whites[i] = 1;
        }
        checkCuda(cudaMemcpy(d_whites, whites, subLen * sizeof(int), cudaMemcpyHostToDevice), "reset whites copy failed");
        updateSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, temp, d_state);
        checkCuda(cudaGetLastError(), "update black launch failed");
        checkCuda(cudaDeviceSynchronize(), "update black execution failed");
        checkCuda(cudaMemcpy(blacks, d_blacks, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy blacks to host failed");
        for(int i = 0; i < subLen; i++){
            if(blacks[i] != 1){
                cerr << "found err black at index " << i << " " << blacks[i] << endl;
                return -1;
            }
        }
    }

    //cleanup
    cudaFree(d_state);
    cudaFree(d_whites);
    cudaFree(d_blacks);
    delete[] whites;
    delete[] blacks;
    return 0;
}