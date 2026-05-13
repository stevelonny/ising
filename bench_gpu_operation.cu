/* benchmark GPU operations, without output/copy obs */
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <ctime>
#include <string>
#include <cstdio>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include "ising_gpu.hpp"

using namespace std;

int main(int argc, char** argv){

    if(argc != 2){ 
        cerr << "Usage: " << argv[0] << " <power of 2>" << endl;
        return -1;
    }
    int power2 = stoi(argv[1]);
    if(power2 <= 0){
        cerr << "Power of 2 for grid size must be a positive integer." << endl;
        return -1;
    }
    int size = 1 << power2; // grid size is 2^power2
    fmt::println("Grid size: {}x{}", size, size);
    int subSize = size/2; //numbers of columns for each sublattice
    int subLen = size*subSize; //length of each sublattice array (total number of elements)
    int Len = 2*subLen; //total number of elements in the lattice
    int nsweeps = 1000;
    float temp_low = 0.8f;
    float temp_high = 3.0f;
    int deltaE[5] = {-8, -4, 0, 4, 8};
    float expTable[5];
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_low);
    }
    expTable[2] = 1.0f;
    
    /* timing setup */
    //start timing for allocating and initializing arrays on GPU
    auto start_init = std::chrono::high_resolution_clock::now();
    //arrays for storing white and black sublattices
    int* d_lattice;
    cudaMalloc(&d_lattice, Len * sizeof(int));
    int* d_whites = d_lattice;
    int* d_blacks = d_lattice + subLen;
    //determine grid and block dimensions
    int nthreads = subLen;
    if(subLen > 1024) nthreads = 1024;
    int nblocks = (subLen + nthreads - 1) / nthreads;
    fmt::println("Launching kernel with {} blocks and {} threads per block.", nblocks, nthreads);
    //init curand
    curandStatePhilox4_32_10_t* d_state;
    cudaMalloc(&d_state, nblocks * nthreads * sizeof(curandStatePhilox4_32_10_t));
    initCurand<<<nblocks, nthreads>>>(d_state, 42);
    //initialize sublattices on GPU
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    int* d_magnetization;
    cudaMalloc(&d_magnetization, nsweeps * sizeof(int));
    cudaMemset(d_magnetization, 0, nsweeps * sizeof(int));
    auto end_init = std::chrono::high_resolution_clock::now();
    auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start_init);

    /* interlude */
    //asd
    
    /* benchmarking (no copy/magnetization) */
    //OBC at low temp
    auto start_obc_low = std::chrono::high_resolution_clock::now();
    for(int s{0}; s < nsweeps; s++){
        updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
        updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
    }
    cudaDeviceSynchronize();
    auto end_obc_low = std::chrono::high_resolution_clock::now();
    auto duration_obc_low = std::chrono::duration_cast<std::chrono::milliseconds>(end_obc_low - start_obc_low);
    //reset lattice
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();
    //OBC at high temp
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_high);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    auto start_obc_high = std::chrono::high_resolution_clock::now();
    for(int s{0}; s < nsweeps; s++){
        updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
        updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
    }
    cudaDeviceSynchronize();
    auto end_obc_high = std::chrono::high_resolution_clock::now();
    auto duration_obc_high = std::chrono::duration_cast<std::chrono::milliseconds>(end_obc_high - start_obc_high);
    //reset lattice
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();
    //PBC at low temp
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_low);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    auto start_pbc_low = std::chrono::high_resolution_clock::now();
    for(int s{0}; s < nsweeps; s++){
        updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
        updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
    }
    cudaDeviceSynchronize();
    auto end_pbc_low = std::chrono::high_resolution_clock::now();
    auto duration_pbc_low = std::chrono::duration_cast<std::chrono::milliseconds>(end_pbc_low - start_pbc_low);
    //reset lattice
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();
    //PBC at high temp
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_high);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    auto start_pbc_high = std::chrono::high_resolution_clock::now();
    for(int s{0}; s < nsweeps; s++){
        updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
        updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
    }
    cudaDeviceSynchronize();
    auto end_pbc_high = std::chrono::high_resolution_clock::now();
    auto duration_pbc_high = std::chrono::duration_cast<std::chrono::milliseconds>(end_pbc_high - start_pbc_high);
    //reset lattice
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();

    /* benchamrking (w/magnetization) */
    DataBlock d;
    d.lattice = d_lattice;
    d.whites = d_whites;
    d.blacks = d_blacks;
    d.size = size;
    d.Len = Len;
    d.subSize = subSize;
    d.subLen = subLen;
    //d.temp = temp;
    d.state = d_state;
    d.magnetization = d_magnetization;
    //OBC at low temp
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_low);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    auto start_obc_low_mag = std::chrono::high_resolution_clock::now();
    blockSweepKernel<Open>(&d, nsweeps, nblocks, nthreads);
    cudaDeviceSynchronize();
    auto end_obc_low_mag = std::chrono::high_resolution_clock::now();
    auto duration_obc_low_mag = std::chrono::duration_cast<std::chrono::milliseconds>(end_obc_low_mag - start_obc_low_mag);
    //reset lattice
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();
    //OBC at high temp
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_high);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    auto start_obc_high_mag = std::chrono::high_resolution_clock::now();
    blockSweepKernel<Open>(&d, nsweeps, nblocks, nthreads);
    cudaDeviceSynchronize();
    auto end_obc_high_mag = std::chrono::high_resolution_clock::now();
    auto duration_obc_high_mag = std::chrono::duration_cast<std::chrono::milliseconds>(end_obc_high_mag - start_obc_high_mag);
    //reset lattice
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();
    //PBC at low temp
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_low);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    auto start_pbc_low_mag = std::chrono::high_resolution_clock::now();
    blockSweepKernel<Periodic>(&d, nsweeps, nblocks, nthreads);
    cudaDeviceSynchronize();
    auto end_pbc_low_mag = std::chrono::high_resolution_clock::now();
    auto duration_pbc_low_mag = std::chrono::duration_cast<std::chrono::milliseconds>(end_pbc_low_mag - start_pbc_low_mag);
    //reset lattice
    initSublatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state);
    cudaDeviceSynchronize();
    //PBC at high temp
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp_high);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    auto start_pbc_high_mag = std::chrono::high_resolution_clock::now();
    blockSweepKernel<Periodic>(&d, nsweeps, nblocks, nthreads);
    cudaDeviceSynchronize();
    auto end_pbc_high_mag = std::chrono::high_resolution_clock::now();
    auto duration_pbc_high_mag = std::chrono::duration_cast<std::chrono::milliseconds>(end_pbc_high_mag - start_pbc_high_mag);

    string filename = fmt::format("gpu_bench_{0}.txt", size);
    //if file already exists, append to it; otherwise, create new file with headers
    {
        std::ifstream infile(filename);
        bool exists = infile.good();
        infile.close();
        std::ofstream outfile(filename, std::ios::app);
        if(!exists){
            fmt::print(outfile, "size\tinit_us\tobc_low_ms\tobc_high_ms\tpbc_low_ms\tpbc_high_ms\tobc_low_mag\tobc_high_mag\tpbc_low_mag\tpbc_high_mag\n");
        }
        // append a summary line (some durations are microseconds, others milliseconds)
        fmt::print(outfile, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                   size,
                   duration_init.count(),
                   duration_obc_low.count(),
                   duration_obc_high.count(),
                   duration_pbc_low.count(),
                   duration_pbc_high.count(),
                   duration_obc_low_mag.count(),
                   duration_obc_high_mag.count(),
                   duration_pbc_low_mag.count(),
                   duration_pbc_high_mag.count());
        outfile.close();
    }


    //cleanup
    cudaFree(d_lattice);
    cudaFree(d_magnetization);
    cudaFree(d_state);

    return 0;
}