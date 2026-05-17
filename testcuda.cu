#include "ising_gpu.hpp"
#include "ising_shared.hpp"
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>

using namespace std;

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fmt::println("Error: {} - {}", msg, cudaGetErrorString(err));
        exit(-1);
    }
}


int main(int argc, char** argv){

    if(argc != 2){
        fmt::println("Usage: {} <grid_size>", argv[0]);
        return -1;
    }
    int size = stoi(argv[1]);
    int subSize = size / 2;
    int subLen = size*subSize;
    int Len = size*size;

    //prepare whites up blacks down
    fmt::println("Testing original checkerboard pattern");
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

    int nthreads = subLen;
    if(subLen > 1024) nthreads = 1024;
    int nblocks = (subLen + nthreads - 1) / nthreads;
    fmt::println("Launching kernel with {} blocks and {} threads per block.", nblocks, nthreads);
    //initialize curand states
    curandStatePhilox4_32_10_t* d_state;
    checkCuda(cudaMalloc(&d_state, subLen * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_state old failed");
    int rd{42};
    initCurand<<<nblocks, nthreads>>>(d_state, rd);
    cudaDeviceSynchronize();
    //low temp/botchered update to verify
    float expTable[5] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f}; // always flip for testing
    checkCuda(cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float)), "copy expTable failed");
    fmt::println("  Testing PERIODIC boundary condition");
    fmt::println("    Updating white sublattice");
    updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
    checkCuda(cudaGetLastError(), "update white launch failed");
    checkCuda(cudaDeviceSynchronize(), "update white execution failed");
    checkCuda(cudaMemcpy(whites, d_whites, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy whites to host failed");
    for(int i = 0; i < subLen; i++){
        if(whites[i] != -1){
            fmt::println("found err white at index {} {}", i, whites[i]);
            return -1;
        }
    }

    for(int i = 0; i < subLen; i++){
        whites[i] = 1;
    }
    checkCuda(cudaMemcpy(d_whites, whites, subLen * sizeof(int), cudaMemcpyHostToDevice), "reset whites copy failed");
    fmt::println("    Updating black sublattice");
    updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
    checkCuda(cudaGetLastError(), "update black launch failed");
    checkCuda(cudaDeviceSynchronize(), "update black execution failed");
    checkCuda(cudaMemcpy(blacks, d_blacks, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy blacks to host failed");
    for(int i = 0; i < subLen; i++){
        if(blacks[i] != 1){
            fmt::println("found err black at index {} {}", i, blacks[i]);
            return -1;
        }
    }
    fmt::println("    PERIODIC: passed");
    //test OPEN boundary condition for sublattice updates
    fmt::println("  Testing OPEN boundary condition");
    fmt::println("    Updating white sublattice");

    for(int i = 0; i < subLen; i++){
        whites[i] = 1;
        blacks[i] = -1;
    }
    checkCuda(cudaMemcpy(d_whites, whites, subLen * sizeof(int), cudaMemcpyHostToDevice), "reset whites copy failed");
    checkCuda(cudaMemcpy(d_blacks, blacks, subLen * sizeof(int), cudaMemcpyHostToDevice), "reset blacks copy failed");

    updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
    checkCuda(cudaGetLastError(), "update white open launch failed");
    checkCuda(cudaDeviceSynchronize(), "update white open execution failed");
    checkCuda(cudaMemcpy(whites, d_whites, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy whites open to host failed");
    for(int i = 0; i < subLen; i++){
        if(whites[i] != -1){
            fmt::println("found err white (open) at index {} {}", i, whites[i]);
            return -1;
        }
    }

    for(int i = 0; i < subLen; i++){
        whites[i] = 1;
    }
    checkCuda(cudaMemcpy(d_whites, whites, subLen * sizeof(int), cudaMemcpyHostToDevice), "reset whites open copy failed");
    fmt::println("    Updating black sublattice");
    updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
    checkCuda(cudaGetLastError(), "update black open launch failed");
    checkCuda(cudaDeviceSynchronize(), "update black open execution failed");
    checkCuda(cudaMemcpy(blacks, d_blacks, subLen * sizeof(int), cudaMemcpyDeviceToHost), "copy blacks open to host failed");
    for(int i = 0; i < subLen; i++){
        if(blacks[i] != 1){
            fmt::println("found err black (open) at index {} {}", i, blacks[i]);
            return -1;
        }
    }
    fmt::println("    OPEN: passed");

    checkCuda(cudaFree(d_whites), "cudaFree d_whites failed");
    checkCuda(cudaFree(d_blacks), "cudaFree d_blacks failed");
    checkCuda(cudaFree(d_state), "cudaFree d_state failed");

    /* ----- */
    //test updateTileKernel
    fmt::println("Testing shared memory update");
    int* lattice = new int[Len];
    int* d_lattice;
    checkCuda(cudaMalloc(&d_lattice, Len * sizeof(int)), "cudaMalloc d_lattice failed");

    int blockDimX = 8;
    int blockDimY = 16;
    int n_TilesRow = size / (blockDimX*2); //number of tiles per row (each tile is blockDimX*2 pixels wide)
    int n_tilesOfOneColor = (n_TilesRow/2) * n_TilesRow; //tiles to process for one color
    int tileWidth = blockDimX*2 + 2; //tile size in pixels (halo added in kernel)
    int tileHeight = blockDimY + 2; //tile size in pixels (halo added in kernel)
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim(n_tilesOfOneColor);
    int sharedMemSize = (tileWidth*tileHeight) * sizeof(int);
    
    auto isValidLattice = [&](const char* stage){
        checkCuda(cudaMemcpy(lattice, d_lattice, Len * sizeof(int), cudaMemcpyDeviceToHost), "copy lattice to host");
        for(int i = 0; i < Len; i++){
            if(lattice[i] != -1 && lattice[i] != 1){
                fmt::println("Invalid spin at index {}: {} ({})", i, lattice[i], stage);
                return false;
            }
        }
        return true;
    };

    int tileStateCount = subLen;
    nthreads = tileStateCount;
    if(tileStateCount > 1024) nthreads = 1024;
    nblocks = (tileStateCount + nthreads - 1) / nthreads;
    curandStatePhilox4_32_10_t* d_state_tile;
    checkCuda(cudaMalloc(&d_state_tile, tileStateCount * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_state_tile failed");
    initCurand<<<nblocks, nthreads>>>(d_state_tile, 42);
    checkCuda(cudaGetLastError(), "init tile state launch failed");
    checkCuda(cudaDeviceSynchronize(), "init tile state execution failed");

    //expTable already set above, just copy again to be sure
    //checkCuda(cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float)), "copy expTable failed");

    //test PERIODIC boundary condition
    fmt::println("  Testing PERIODIC boundary condition");
    
    //init lattice
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            lattice[i * size + j] = ((i + j) % 2 == 0) ? 1 : -1;
        }
    }
    checkCuda(cudaMemcpy(d_lattice, lattice, Len * sizeof(int), cudaMemcpyHostToDevice), "copy lattice to device");

    //update red tiles (color = 0)
    fmt::println("    Updating red tiles...");
    updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, d_state_tile);
    checkCuda(cudaGetLastError(), "update red tiles launch failed");
    checkCuda(cudaDeviceSynchronize(), "update red tiles execution failed");
    if(!isValidLattice("after red tile update (Periodic)")) return -1;

    //update blue tiles (color = 1)
    fmt::println("    Updating blue tiles...");
    updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, d_state_tile);
    checkCuda(cudaGetLastError(), "update blue tiles launch failed");
    checkCuda(cudaDeviceSynchronize(), "update blue tiles execution failed");
    if(!isValidLattice("after blue tile update (Periodic)")) return -1;
    fmt::println("    PERIODIC: passed");

    //test OPEN boundary condition
    fmt::println("  Testing OPEN boundary condition");
    
    // reset lattice
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            lattice[i * size + j] = ((i + j) % 2 == 0) ? 1 : -1;
        }
    }
    checkCuda(cudaMemcpy(d_lattice, lattice, Len * sizeof(int), cudaMemcpyHostToDevice), "copy lattice to device");

    //update red tiles (color = 0)
    fmt::println("    Updating red tiles...");
    updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, d_state_tile);
    checkCuda(cudaGetLastError(), "update red tiles open launch failed");
    checkCuda(cudaDeviceSynchronize(), "update red tiles open execution failed");
    if(!isValidLattice("after red tile update (Open)")) return -1;

    //update blue tiles (color = 1)
    fmt::println("    Updating blue tiles...");
    updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, d_state_tile);
    checkCuda(cudaGetLastError(), "update blue tiles open launch failed");
    checkCuda(cudaDeviceSynchronize(), "update blue tiles open execution failed");
    if(!isValidLattice("after blue tile update (Open)")) return -1;
    fmt::println("    OPEN: passed");

    //cleanuptest
    checkCuda(cudaFree(d_state_tile), "cudaFree d_state_tile failed");
    checkCuda(cudaFree(d_lattice), "cudaFree d_lattice failed");
    delete[] lattice;
    delete[] whites;
    delete[] blacks;
    return 0;
}