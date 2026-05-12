#include "ising_gpu.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <fmt/format.h>
#include <fmt/ostream.h>
//#include <algorithm>

using namespace std;

#define MAX_SWEEPS 10000

double compute_mean(int* data, int n){
    double mean{0.};
    for(int i{0}; i<n; i++){
        mean += data[i];
    }
    return mean / (double)n;
}

double compute_mean(double* data, int n){
    double mean{0.};
    for(int i{0}; i<n; i++){
        mean += data[i];
    }
    return mean / (double)n;
}

double compute_var(int* data, int n, double mean){
    double var{0.};
    for(int i{0}; i<n; i++){
        var += (data[i] - mean) * (data[i] - mean);
    }
    return var / (double)n;
}

int main(int argc, char** argv){

    if(argc != 6 && argc != 7){ 
        fmt::print(stderr, "Usage: {} <power of 2 for grid_size> <temperature> <nblocksweeps> <blocksize> <obc|pbc> <run_tag>\n", argv[0]);
        return -1;
    }
    int power2 = stoi(argv[1]);
    if(power2 <= 0){
        fmt::print(stderr, "Power of 2 for grid size must be a positive integer.\n");
        return -1;
    }
    double temp = stod(argv[2]);
    if(temp <= 0.0){
        fmt::print(stderr, "Temperature must be positive.\n");
        return -1;
    }
    int nblocksweeps = stoi(argv[3]);
    if(nblocksweeps <= 0){
        fmt::print(stderr, "Number of blocksweeps must be positive.\n");
        return -1;
    }
    int blocksize = stoi(argv[4]);
    if(blocksize <= 0){
        fmt::print(stderr, "Block size must be positive.\n");
        return -1;
    }
    bool obc = false;
    
    if(string(argv[5]) == "obc"){
        obc = true;
        fmt::println("Using open boundary conditions.");
    }
    else if(string(argv[5]) == "pbc"){
        obc = false;
        fmt::println("Using periodic boundary conditions.");
    }
    else{
        fmt::print(stderr, "Invalid boundary condition. Use 'obc' or 'pbc'.\n");
        return -1;
    }
    if(argc == 7){
        fmt::println("Run tag: {}", argv[6]);
    }
    //run_tag used for distinguishing different runs, can be used in filename or just printed out
    //default is empty string
    string run_tag = (argc == 7) ? argv[6] : "";

    /* setup */
    int size = 1 << power2; // grid size is 2^power2
    fmt::println("Grid size: {}x{}", size, size);
    int subSize = size/2; //numbers of columns for each sublattice
    int subLen = size*subSize; //length of each sublattice array (total number of elements)
    int Len = 2*subLen; //total number of elements in the lattice
    //determine grid and block dimensions
    int nthreads = subLen;
    if(subLen > 1024) nthreads = 1024;
    int nblocks = (subLen + nthreads - 1) / nthreads;
    fmt::println("Launching kernel with {} blocks and {} threads per block.", nblocks, nthreads);
    //allocate arrays on gpu
    //allocate a contiguous block for both sublattices (useful for thrust reduction)
    int* d_lattice;
    cudaMalloc(&d_lattice, 2 * subLen * sizeof(int));
    int* d_whites = d_lattice;
    int* d_blacks = d_lattice + subLen;
    int deltaE[5] = {-8, -4, 0, 4, 8};
    float expTable[5];

    //init curand state
    std::random_device rd;
    //int rd{42};
    curandStatePhilox4_32_10_t* d_state;
    cudaMalloc(&d_state, nblocks * nthreads * sizeof(curandStatePhilox4_32_10_t));
    initCurand<<<nblocks, nthreads>>>(d_state, rd());
    //initialize sublattices on gpu
    initSublatticeKernel<<<nblocks, nthreads>>>(d_whites, subLen, d_state);
    initSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, subLen, d_state);
    cudaDeviceSynchronize();

    int sweep{0};
    int index{0};
    int window_size{500};
    int* d_magnetization;
    cudaMalloc(&d_magnetization, window_size * sizeof(int));
    cudaMemset(d_magnetization, 0, window_size * sizeof(int));
    int* magnetization = new int[window_size];
    
    /* annealing */
    //useful when using open border condition
    if(true/* obc */){ //temperature annealing schedule: cool from high T to target over 1000 sweeps
        int anneal_sweeps = 1000;
        double T_start{5.0};
        double T_current{5.0};
        fmt::println("Annealing phase (annealing T: {} -> {})", T_start, temp);
        while(sweep < anneal_sweeps){
            T_current = T_start - (T_start - temp) * ((double)sweep / (double)(anneal_sweeps - 1));
            for(int i{0}; i<5; i++){
                expTable[i] = exp(-(float)deltaE[i] / T_current);
            }
            expTable[2] = 1.0; //exp(0) is always 1
            cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
            //perform mc sweep
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
            //compute magnetization (debug/logging)
            //index = sweep%window_size;
            //cudaMemset(&d_magnetization[index], 0, sizeof(int));
            //reduceMagnetizationKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, subLen, &d_magnetization[index]);

            /* if(sweep%100 == 0){
                cudaMemcpy(&magnetization[index], &d_magnetization[index], sizeof(int), cudaMemcpyDeviceToHost);
                fmt::println("Sweep {:<6} T: {:<6.1f} Mag: {:<10}", sweep, T_current, magnetization[index]);
            } */

            sweep++;
        }
        //cudaMemcpy(&magnetization[index], &d_magnetization[index], sizeof(int), cudaMemcpyDeviceToHost);
        //fmt::println("Sweep {:<6} T: {:<6.1f} Mag: {:<10}", sweep, T_current, magnetization[index]);
        fmt::println("Annealing completed. Resetting sweep counter.");
        sweep = 0;
    }
    
    /* thermalization */
    //check convergence only after cooling completes
    double mean1{0.};
    double mean2{0.};
    double var1{0.};
    double var2{0.};
    bool is_thermalized{false};
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp);
    }
    expTable[2] = 1.0; //exp(0) is always 1
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    fmt::println("Thermalization phase");
    while(!is_thermalized && sweep < MAX_SWEEPS){
        //perform mc sweep
        if(obc){
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
        }
        else{
            updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, d_state);
            updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, d_state);
        }
        //compute magnetization
        index = sweep%window_size;
        cudaMemset(&d_magnetization[index], 0, sizeof(int));
        reduceMagnetizationThrust<<<nblocks, nthreads>>>(d_lattice, Len, &d_magnetization[index]);

        if(sweep>=window_size && sweep%100 == 0){
            //retrieve d_magnetization
            cudaMemcpy(magnetization, d_magnetization, window_size * sizeof(int), cudaMemcpyDeviceToHost);
            //calculate mean and var for 2 consecutive subwindows
            mean1 = 0.;
            mean2 = 0.;
            var1 = 0.;
            var2 = 0.;
            for(int i{0}; i<window_size/2; i++){
                mean1 += magnetization[i];
                mean2 += magnetization[i + window_size/2];
            }
            mean1 /= (double)window_size/2.;
            mean2 /= (double)window_size/2.;
            for(int i{0}; i<window_size/2; i++){
                var1 += (magnetization[i] - mean1) * (magnetization[i] - mean1);
                var2 += (magnetization[i + window_size/2] - mean2) * (magnetization[i + window_size/2] - mean2);
            }
            var1 /= (double)window_size/2.;
            var2 /= (double)window_size/2.;
            //diff and threshold
            double diff = fabs(mean1 - mean2);
            double threshold = max(0.05 * sqrt(var1 + var2), 1e-6);
            if(diff < threshold /* && (var1 > 1.0 || var2 > 1.0) */){//zero variance can mean frozen system
                is_thermalized = true;
                fmt::println("Thermalization achieved at sweep {}", sweep);
                if(diff < 1e-6){
                    fmt::println("Very low variance detected. diff: {:<10.6f} threshold: {:<10.6f}", diff, threshold);
                }
                break;
            }
        }
        sweep++;
    }
    fmt::println("Final thermalization: {}", (is_thermalized ? "SUCCESS" : "INCOMPLETE"));
    //fmt::println("values: {}±{}\t{}±{} diff: {:<10.6f} threshold: {:10.6f}", mean1, sqrt(var1), mean2, sqrt(var2), fabs(mean1-mean2), min(0.05 * sqrt(var1 + var2), 1e-5));

    /* interlude */
    cudaFree(d_magnetization);
    DataBlock d;
    d.lattice = d_lattice;
    d.whites = d_whites;
    d.blacks = d_blacks;
    d.size = size;
    d.Len = Len;
    d.subSize = subSize;
    d.subLen = subLen;
    d.temp = temp;
    d.state = d_state;
    cudaMalloc(&d.magnetization, blocksize * sizeof(int));
    cudaMemset(d.magnetization, 0, blocksize * sizeof(int));
    //open fileout
    string filename = fmt::format("size_{}_b{}_n{}_bc_{}{}.dat", size, nblocksweeps, blocksize, (obc ? "obc" : "pbc"), "_run" + run_tag);
    fmt::print("Saving to {}\n", filename);
    ofstream fileout;
    fileout.open(filename);

    /* simulation/measure */
    //may need just one block but oh well
    for(int b{0}; b<nblocksweeps; b++){
        //single block sweep
        if(obc){
            blockSweepKernel<Open>(&d, blocksize, nblocks, nthreads);
        }
        else{
            blockSweepKernel<Periodic>(&d, blocksize, nblocks, nthreads);
        }
        //copy magnetization values
        cudaMemcpy(magnetization, d.magnetization, blocksize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(d.magnetization, 0, blocksize * sizeof(int));
        double mean = compute_mean(magnetization, blocksize);
        double var = compute_var(magnetization, blocksize, mean);

        fmt::println("Block #{:<6} Mag: {:<15.2f} std: {:<10.2f}", b, mean, sqrt(var));
        fmt::print(fileout, "{:<3.1f} {:<6} {:<15.2f} {:<10.2f}\n", temp, b, mean, sqrt(var));
    }

    /* cleanup */
    cudaFree(d_lattice);
    cudaFree(d_magnetization);
    cudaFree(d_state);
    delete[] magnetization;

    return 0;
}