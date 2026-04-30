/* benchmark GPU, without output/copy obs */
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <ctime>
#include "ising_gpu.hpp"

using namespace std;

int main(int argc, char** argv){

    if(argc != 4){ 
        cerr << "Usage: " << argv[0] << " <grid_size> <temperature> <nsweeps>" << endl;
        return -1;
    }
    int size = stoi(argv[1]);
    if (size % 2 != 0 || size <= 0) {
        cerr << "Grid size must be a positive even integer." << endl;
        return -1;
    }
    double temp = stod(argv[2]);
    if (temp <= 0.0) {
        cerr << "Temperature must be positive." << endl;
        return -1;
    }
    int nsweeps = stoi(argv[3]);
    if (nsweeps <= 0) {
        cerr << "Number of sweeps must be positive." << endl;
        return -1;
    }

    int subSize = size/2; //numbers of columns for each sublattice
    int subLen = size*subSize; //length of each sublattice array (total number of elements)
    
    //start timing for allocating and initializing arrays on GPU
    auto start_init = std::chrono::high_resolution_clock::now();
    
    //arrays for storing white and black sublattices
    int* d_whites;
    int* d_blacks;
    cudaMalloc(&d_whites, subLen * sizeof(int));
    cudaMalloc(&d_blacks, subLen * sizeof(int));

    //determine grid and block dimensions
    int nthreads = subLen;
    if(subLen > 1024) nthreads = 1024;
    int nblocks = (subLen + nthreads - 1) / nthreads;
    cout << "Launching kernel with " << nblocks << " blocks and " << nthreads << " threads per block." << endl;
    
    //init curand
    curandState* d_state;
    cudaMalloc(&d_state, nblocks * nthreads * sizeof(curandState));
    initCurand<<<nblocks, nthreads>>>(d_state, 42);

    //initialize sublattices on GPU
    initSublatticeKernel<<<nblocks, nthreads>>>(d_whites, subLen, d_state);
    initSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, subLen, d_state);
    cudaDeviceSynchronize();

    auto end_init = std::chrono::high_resolution_clock::now();
    auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start_init);

    //run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for(int s{0}; s < nsweeps; s++){
        updateSublatticeKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, temp, d_state);
        updateSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, temp, d_state);
    }
    
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    ofstream outFile("results_gpu.txt", ios::app);
    outFile << size << "\t" << temp << "\t" << nsweeps << "\t" << duration_init.count() << "\t" << duration.count() << endl;
    outFile.close();

    cout << "Size" << "\t" << "temp" << "\t" << "nsweeps" << "\t" << "Init" << "\t" << "Sim" << endl;
    cout << size << "\t" << temp << "\t" << nsweeps << "\t" << duration_init.count() << "\t" << duration.count() << endl;

    //cleanup
    cudaFree(d_whites);
    cudaFree(d_blacks);
    cudaFree(d_state);

    return 0;
}