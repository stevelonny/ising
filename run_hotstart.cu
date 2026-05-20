#include "ising_shared.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <fmt/format.h>
#include <fmt/ostream.h>

int main(int argc, char** argv){
    if(argc != 5 && argc != 6){ 
        fmt::print(stderr, "Usage: {} <power of 2 for grid_size> <temperature> <nsweeps> <obc|pbc> [run_tag]\n", argv[0]);
        return -1;
    }
    int power2 = stoi(argv[1]);
    if(power2 < 5){
        fmt::print(stderr, "Power of 2 for grid size must be a positive integer greater than or equal to 5.\n");
        return -1;
    }
    double temp = stod(argv[2]);
    if(temp <= 0.0){
        fmt::print(stderr, "Temperature must be positive.\n");
        return -1;
    }
    int nsweeps = stoi(argv[3]);
    if(nsweeps <= 0){
        fmt::print(stderr, "Number of sweeps must be positive.\n");
        return -1;
    }
    bool obc = false;
    if(string(argv[4]) == "obc"){
        obc = true;
        fmt::println("Using open boundary conditions.");
    }
    else if(string(argv[4]) == "pbc"){
        obc = false;
        fmt::println("Using periodic boundary conditions.");
    }
    else{
        fmt::print(stderr, "Invalid boundary condition. Use 'obc' or 'pbc'.\n");
        return -1;
    }
    if(argc == 6){
        fmt::println("Run tag: {}", argv[5]);
    }
    //run_tag used for distinguishing different runs, can be used in filename or just printed out
    //default is empty string
    string run_tag = (argc == 6) ? argv[5] : "_default";

    /* setup */
    int size = 1 << power2;
    int subSize = size/2;
    int Len = size * size;
    int subLen = subSize * subSize;
    int* d_lattice;
    cudaMalloc(&d_lattice, Len * sizeof(int));
    const int blockDimX = 8;
    const int blockDimY = 16;
    int n_TilesRow = size / (blockDimX*2); //number of tiles per row (each tile is blockDimX*2 pixels wide)
    int n_tilesOfOneColor = (n_TilesRow/2) * n_TilesRow; //tiles to process for one color
    int tileWidth = blockDimX*2 + 2; //tile size in pixels (halo added in kernel)
    int tileHeight = blockDimY + 2; //tile size in pixels (halo added in kernel)
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim(n_tilesOfOneColor);
    int sharedMemSize = (tileWidth*tileHeight) * sizeof(int);

    int tileStateCount = Len / 2;
    int nthreads = tileStateCount;
    if(tileStateCount > 1024) nthreads = 1024;
    int nblocks = (tileStateCount + nthreads - 1) / nthreads;
    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, tileStateCount * sizeof(curandStatePhilox4_32_10_t));
    std::random_device rd;
    initCurand<<<nblocks, nthreads>>>(d_states, rd());
    cudaDeviceSynchronize();
    initLatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_states);
    cudaDeviceSynchronize();

    int deltaE[5] = {-8, -4, 0, 4, 8};
    float expTable[5];
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp);
    }
    expTable[2] = 1.0f; //exp(0) is always 1
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    /* thermalization */
    fmt::println("Thermalization phase");
    int thermalSweeps{10000};
    if(obc){
        for(int s{0}; s<thermalSweeps; s++){
            updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, d_states);
            updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, d_states);
        }
    }
    else{
        for(int s{0}; s<thermalSweeps; s++){
            updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, d_states);
            updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, d_states);
        }
    }
    /* interlude */
    DataBlockShared d;
    d.lattice = d_lattice;
    d.size = size;
    d.Len = Len;
    d.subLen = subLen;
    d.subSize = subSize;
    d.state = d_states;
    d.temp = temp;
    if(cudaMalloc(&d.magnetization, nsweeps * sizeof(int)) != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed!\n");
    }
    if(cudaMemset(d.magnetization, 0, nsweeps * sizeof(int)) != cudaSuccess) {
        fmt::print(stderr, "cudaMemset failed!\n");
    }
    int* magnetization = new int[nsweeps];
    //open fileout
    string filename = fmt::format("size_{}_temp{:.2f}_ns{}_bc_{}{}.dat", size, temp, nsweeps, (obc ? "obc" : "pbc"), "_run" + run_tag);
    fmt::print("Saving to {}\n", filename);
    ofstream fileout;
    fileout.open(filename);

    /* simulation/measure */
    fmt::println("Measurement phase");
    auto start = std::chrono::high_resolution_clock::now();
    if(obc){
        launchCooperativeUpdateTileKernel<Open>(&d, nsweeps);
    }
    else{
        launchCooperativeUpdateTileKernel<Periodic>(&d, nsweeps);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    fmt::println("Measurement completed in {} μs.", duration);
    fmt::println("Measurement completed. Retrieving data...");
    //retrieve d_magnetization
    if(cudaMemcpy(magnetization, d.magnetization, nsweeps * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed!\n");
    }
    fmt::println("Data retrieval completed. Saving to file...");
    for(int i{0}; i<nsweeps; i++){
        //fileout << magnetization[i] << "\n";
        fmt::println(fileout, "{}", magnetization[i]);
    }
    fileout.close();
    fmt::println("Data saved to file.");
    
    /* cleanup */
    delete[] magnetization;
    cudaFree(d.lattice);
    cudaFree(d.state);
    cudaFree(d.magnetization);


    return 0;
}