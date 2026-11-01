#include "ising_shared.hpp"
#include "observables.hpp"
#include "thermalization.hpp"
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
    DataBlockShared d;
    int size = 1 << power2;
    int subSize = size/2;
    int Len = size * size;
    int subLen = subSize * subSize;
    fmt::println(cerr, "Setting up");
    cudaMalloc(&d.lattice, Len * sizeof(int));
    d.size = size;
    d.Len = Len;
    d.subLen = subLen;
    d.subSize = subSize;
    d.temp = temp;
    //curand state init
    int tileStateCount = Len / 2;
    int nthreads = tileStateCount;
    if(tileStateCount > 1024) nthreads = 1024;
    int nblocks = (tileStateCount + nthreads - 1) / nthreads;
    //curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d.state, tileStateCount * sizeof(curandStatePhilox4_32_10_t));
    std::random_device rd;
    initCurand<<<nblocks, nthreads>>>(d.state, 42);
    cudaDeviceSynchronize();

    /* annealing */
    fmt::println(cerr, "Initializing lattice at T = {:.4f}", temp);
    if(temp <= (double)(CRIT_TEMP)){
        initColdLatticeKernel<<<nblocks, nthreads>>>(d.lattice, Len);
        cudaDeviceSynchronize();
        if(obc){
            anneal<Open>(&d, temp);
        }
        else{
            anneal<Periodic>(&d, temp);
        }
    }
    else{
        initLatticeKernel<<<nblocks, nthreads>>>(d.lattice, Len, d.state);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    /* interlude A */
    float expTable[9];
    for(int i{0}; i<9; i++){
        expTable[i] = exp(-(float)_deltaE[i] / temp);
    }
    cudaMemcpyToSymbol(d_expTable, expTable, 9 * sizeof(float));

    /* thermalization */
    ThermalizationResult therm_result;
    if(obc){
        therm_result = thermalize<Open>(&d);
    }
    else{
        therm_result = thermalize<Periodic>(&d);
    }
    cudaDeviceSynchronize();
    bool is_thermalized = therm_result.thermalized;
    fmt::println("Thermalization {} with {} sweeps", is_thermalized, therm_result.sweeps);
    /* interlude  B */
    checkCuda(cudaMalloc(&d.magnetization, nsweeps * sizeof(int)), "Failed to allocate magnetization on device");
    checkCuda(cudaMalloc(&d.energy, nsweeps * sizeof(int)), "Failed to allocate energy on device");
    checkCuda(cudaMemset(d.magnetization, 0, nsweeps * sizeof(int)), "Failed to memset magnetization");
    checkCuda(cudaMemset(d.energy, 0, nsweeps * sizeof(int)), "Failed to memset energy");
    int *magnetization = new int[nsweeps];
    int *energy = new int[nsweeps];

    /* simulation/measure */
    fmt::println("Measurement phase");
    if(obc){
        blockSweepKernelShared<Open>(&d, nsweeps);
    }
    else{
        blockSweepKernelShared<Periodic>(&d, nsweeps);
    }
    fmt::println("Measurement completed. Retrieving data...");
    //retrieve d_mag and d_energy
    checkCuda(cudaMemcpy(magnetization, d.magnetization, nsweeps * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy magnetization from device to host");
    checkCuda(cudaMemcpy(energy, d.energy, nsweeps * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy energy from device to host");
    fmt::println("Data retrieval completed. Processing data...");
    vector<double> M_abs(nsweeps);
    vector<double> M_sq(nsweeps);
    vector<double> M_quad(nsweeps);
    vector<double> E(nsweeps);
    vector<double> E_sq(nsweeps);

    for(int i{0}; i<nsweeps; ++i){
        double m = (double)magnetization[i];
        double m_abs = std::abs(m);
        double m2 = m*m;
        double e = (double)energy[i];
        double e2 = e*e;
        
        M_abs[i] = m_abs;
        M_sq[i] = m2;
        M_quad[i] = m2 * m2;
        E[i] = e;
        E_sq[i] = e2;
    }

    BlockResult res_M_abs = calculateBlockedError(M_abs);
    BlockResult res_M2 = calculateBlockedError(M_sq);
    BlockResult res_M4 = calculateBlockedError(M_quad);
    BlockResult res_E = calculateBlockedError(E);
    BlockResult res_E2 = calculateBlockedError(E_sq);

    fmt::println(std::cout,
            "{:<10}{:<3}{:<10}{:<15}{:<15}{:<15}",
            "obs", "lv", "n_blocks", "mean", "error", "variance");
    printBlockResult(res_M_abs, std::cout, "M_abs");
    printBlockResult(res_M2, std::cout, "M2");
    printBlockResult(res_M4, std::cout, "M4");
    printBlockResult(res_E, std::cout, "E");
    printBlockResult(res_E2, std::cout, "E2");

    std::string filename = "master_results_energy.csv";
    std::ofstream fileout;
    std::ifstream infile(filename);
    bool write_header = !infile.good();
    infile.close();

    fileout.open(filename, std::ios_base::app);
    if(write_header) fmt::println(fileout, "L,T,run_tag,is_thermalized,M_abs,err_M_abs,tau_M_abs,M2,err_M2,tau_M2,M4,err_M4,tau_M4,E,err_E,tau_E,E2,err_E2,tau_E2");
    
    fmt::println(
        fileout,
        "{},{:.4f},{},{},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f}",
        size, temp, run_tag, (is_thermalized ? 1 : 0),
        res_M_abs.mean, res_M_abs.error, res_M_abs.tau_int,
        res_M2.mean, res_M2.error, res_M2.tau_int,
        res_M4.mean, res_M4.error, res_M4.tau_int,
        res_E.mean, res_E.error, res_E.tau_int,
        res_E2.mean, res_E2.error, res_E2.tau_int
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