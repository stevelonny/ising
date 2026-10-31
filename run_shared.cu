#include "ising_shared.hpp"
#include "observables.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <fmt/format.h>
#include <fmt/ostream.h>

#define MAX_SWEEPS 500000

int main(int argc, char** argv){
    if(argc != 5 && argc != 6){ 
        fmt::print(stderr, "Usage: {} <power of 2 for grid_size> <temperature> <nsweeps> <obc|pbc> [run_tag]\n", argv[0]);
        return -1;
    }
    int power2 = stoi(argv[1]);
    if(power2 < 6){
        fmt::print(stderr, "Power of 2 for grid size must be a positive integer greater than or equal to 6.\n");
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
    string run_tag = (argc == 6) ? argv[5] : "tagless";

    /* setup */
    int size = 1 << power2;
    int subSize = size/2;
    int Len = size * size;
    int subLen = subSize * subSize;
    int* d_lattice;
    cudaMalloc(&d_lattice, Len * sizeof(int));
    const int TILE_DIM_X = __TILE_DIM_X__;
    const int TILE_DIM_Y = __TILE_DIM_Y__;
    const int blockDimX = TILE_DIM_X/2;
    const int blockDimY = TILE_DIM_Y;
    int n_TilesRow = size / (TILE_DIM_X); //number of tiles per row (each tile is blockDimX*2 pixels wide or TILE_DIM_X pixels wide)
    int n_tilesOfOneColor = (n_TilesRow/2) * n_TilesRow; //tiles to process for one color
    int tileWidth = TILE_DIM_X + 2; //tile size in pixels (halo added in kernel)
    int tileHeight = TILE_DIM_Y + 2; //tile size in pixels (halo added in kernel)
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

    int sweep{0};
    int index{0};
    int window_size{10000};
    int* d_magnetization;
    int * d_energy;
    cudaMalloc(&d_magnetization, window_size * sizeof(int));
    cudaMalloc(&d_energy, window_size * sizeof(int));
    cudaMemset(d_magnetization, 0, window_size * sizeof(int));
    cudaMemset(d_energy, 0, window_size * sizeof(int));
    int* magnetization = new int[window_size];
    int* energy = new int[window_size];

    /* annealing */
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
            if(obc){
                updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, d_states);
                updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, d_states);
            }
            else{
                updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, d_states);
                updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, d_states);
            }
            sweep++;
        }
        fmt::println("Annealing completed. Resetting sweep counter.");
        sweep = 0;
    }
    /* thermalization */
    double mean1{0.};
    double mean2{0.};
    double var1{0.};
    double var2{0.};
    bool is_thermalized{false};
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temp);
    }
    expTable[2] = 1.0f; //exp(0) is always 1
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));
    fmt::println("Thermalization phase");
    while(!is_thermalized && sweep < MAX_SWEEPS){
        //perform mc sweep
        index = sweep%window_size;
        cudaMemset(&d_magnetization[index], 0, sizeof(int));
        if(obc){
            updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, &d_magnetization[index], d_states);
            updateTileKernel<Open><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, &d_magnetization[index], d_states);
        }
        else{
            updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 0, n_TilesRow, &d_magnetization[index], d_states);
            updateTileKernel<Periodic><<<gridDim, blockDim, sharedMemSize>>>(d_lattice, size, 1, n_TilesRow, &d_magnetization[index], d_states);
        }
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
    
    /* interlude */
    delete[] magnetization;
    magnetization = new int[nsweeps];
    cudaFree(d_magnetization);
    DataBlockShared d;
    d.lattice = d_lattice;
    d.size = size;
    d.Len = Len;
    d.subLen = subLen;
    d.subSize = subSize;
    d.state = d_states;
    d.temp = temp;
    
    checkCuda(cudaMalloc(&d.magnetization, nsweeps * sizeof(int)), "Failed to allocate magnetization on device");
    checkCuda(cudaMemset(d.magnetization, 0, nsweeps * sizeof(int)), "Failed to memset magnetization");

    /* simulation/measure */
    fmt::println("Measurement phase");
    if(obc){
        blockSweepKernelShared<Open>(&d, nsweeps);
    }
    else{
        blockSweepKernelShared<Periodic>(&d, nsweeps);
    }
    fmt::println("Measurement completed. Retrieving data...");
    //retrieve d_magnetization
    checkCuda(cudaMemcpy(magnetization, d.magnetization, nsweeps * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy magnetization from device to host");
    fmt::println("Data retrieval completed. Processing data...");
    vector<double> M_abs(nsweeps);
    vector<double> M_sq(nsweeps);
    vector<double> M_quad(nsweeps);
    double mean_M4 = 0.0;

    for(int i{0}; i<nsweeps; ++i){
        double m = (double)magnetization[i];
        double m_abs = std::abs(m);
        double m2 = m*m;
        
        M_abs[i] = m_abs;
        M_sq[i] = m2;
        M_quad[i] = m2 * m2;
    }

    BlockResult res_M_abs = calculateBlockedError(M_abs, 15);
    BlockResult res_M2 = calculateBlockedError(M_sq, 15);
    BlockResult res_M4 = calculateBlockedError(M_quad, 15);

    std::string filename = "master_results_new.csv";
    std::ofstream fileout;
    std::ifstream infile(filename);
    bool write_header = !infile.good();
    infile.close();

    fileout.open(filename, std::ios_base::app);
    if(write_header) fmt::println(fileout, "L,T,run_tag,is_thermalized,M_abs,err_M_abs,tau_M_abs,M2,err_M2,tau_M2,M4,err_M4,tau_M4");
    
    fmt::println(
        fileout,
        "{},{:.4f},{},{},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f}",
        size, temp, run_tag, (is_thermalized ? 1 : 0),
        res_M_abs.mean, res_M_abs.error, res_M_abs.tau_int,
        res_M2.mean, res_M2.error, res_M2.tau_int,
        res_M4.mean, res_M4.error, res_M4.tau_int
    );
    fileout.close();
    fmt::println("Data saved to {}.", filename);
    
    /* cleanup */
    delete[] magnetization;
    cudaFree(d.lattice);
    cudaFree(d.state);
    cudaFree(d.magnetization);

    return 0;
}