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
#include <getopt.h>

bool g_verbose = false;

template<typename... Args>
void vprint(fmt::format_string<Args...> fmt_str, Args&&... args) {
    if (g_verbose)
        fmt::println(stderr, fmt_str, std::forward<Args>(args)...);
}

static void print_usage(const char* prog) {
    fmt::print(stderr,
        "Usage: {} -L <log2_size> -T <temp> -N <nsweeps> -b <obc|pbc> "
        "[-t <tag>] [-o <output.csv>] [-s <seed>] [-v]\n\n"
        "  -L, --log2-size  Power of 2 for grid size (integer >= 6)\n"
        "  -T, --temp       Temperature (float > 0)\n"
        "  -N, --nsweeps    Number of measurement sweeps (int > 0)\n"
        "  -b, --bc         Boundary conditions: obc or pbc\n"
        "  -t, --tag        Run tag for output CSV  [default: tagless]\n"
        "  -o, --output     Output CSV filename     [default: master_results_energy.csv]\n"
        "  -s, --seed       PRNG seed               [default: 42]\n"
        "  -v, --verbose    Enable verbose output   [default: false]\n",
        prog);
}

int main(int argc, char** argv){
    /* defaults */
    int       power2   = -1;
    double    temp     = -1.0;
    int       nsweeps  = -1;
    bool      obc      = false;
    bool      bc_set   = false;
    string    run_tag  = "tagless";
    string    outfile  = "master_results.csv";
    std::random_device rd;
    unsigned long seed = rd();
    static const struct option long_opts[] = {
        {"log2-size", required_argument, nullptr, 'L'},
        {"temp",      required_argument, nullptr, 'T'},
        {"nsweeps",   required_argument, nullptr, 'N'},
        {"bc",        required_argument, nullptr, 'b'},
        {"tag",       required_argument, nullptr, 't'},
        {"output",    required_argument, nullptr, 'o'},
        {"seed",      required_argument, nullptr, 's'},
        {"verbose",   no_argument,       nullptr, 'v'},
        {"help",      no_argument,       nullptr, 'h'},
        {nullptr,     0,                 nullptr,  0 }
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "L:T:N:b:t:o:s:vh", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'L': power2  = stoi(optarg); break;
            case 'T': temp    = stod(optarg); break;
            case 'N': nsweeps = stoi(optarg); break;
            case 'b':
                if (string(optarg) == "obc")      { obc = true;  bc_set = true; }
                else if (string(optarg) == "pbc") { obc = false; bc_set = true; }
                else {
                    fmt::print(stderr, "Error: --bc must be 'obc' or 'pbc'.\n");
                    return -1;
                }
                break;
            case 't': run_tag = optarg; break;
            case 'o': outfile = optarg; break;
            case 's': seed    = stoul(optarg); break;
            case 'v': g_verbose = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return -1;
        }
    }
    /* validation */
    bool ok = true;
    if (power2 < 6)   { fmt::print(stderr, "Error: -L must be >= 6.\n");   ok = false; }
    if (temp   <= 0)  { fmt::print(stderr, "Error: -T must be > 0.\n");    ok = false; }
    if (nsweeps <= 0) { fmt::print(stderr, "Error: -N must be > 0.\n");    ok = false; }
    if (!bc_set)      { fmt::print(stderr, "Error: -b is required.\n");    ok = false; }
    if (!ok) { print_usage(argv[0]); return -1; }

    /* setup */
    DataBlockShared d;
    int size = 1 << power2;
    int subSize = size/2;
    int Len = size * size;
    int subLen = subSize * subSize;
    vprint("Setting up");
    cudaMalloc(&d.lattice, Len * sizeof(int));
    d.size = size;
    d.Len = Len;
    d.subLen = subLen;
    d.subSize = subSize;
    d.temp = temp;
    vprint("Allocated lattice of size {}x{} (Len = {}, subLen = {})", size, size, Len, subLen);
    //curand state init
    int tileStateCount = Len / 2;
    int nthreads = tileStateCount;
    if(tileStateCount > 1024) nthreads = 1024;
    int nblocks = (tileStateCount + nthreads - 1) / nthreads;
    //curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d.state, tileStateCount * sizeof(curandStatePhilox4_32_10_t));
    initCurand<<<nblocks, nthreads>>>(d.state, seed);
    cudaDeviceSynchronize();
    vprint("Initialized curand states with seed\n{}", seed);

    /* annealing */
    vprint("Temperature: T = {:.4f}", temp);
    if(temp <= (double)(CRIT_TEMP)){
        vprint("Cold start: T <= Tc, initializing all spins up and annealing", temp);
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
        vprint("Hot start: T > Tc, initializing random spins", temp);
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
    vprint("Starting thermalization");
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
    vprint("Measurement phase");
    if(obc){
        blockSweepKernelShared<Open>(&d, nsweeps);
    }
    else{
        blockSweepKernelShared<Periodic>(&d, nsweeps);
    }
    vprint("Measurement completed. Retrieving data...");
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

    std::string filename = outfile;
    std::ofstream fileout;
    std::ifstream infile(filename);
    bool write_header = !infile.good();
    infile.close();

    fileout.open(filename, std::ios_base::app);
    if(write_header) fmt::println(fileout, "L,T,run_tag,is_thermalized,M_abs,err_M_abs,tau_M_abs,M2,err_M2,tau_M2,M4,err_M4,tau_M4,E,err_E,tau_E,E2,err_E2,tau_E2,seed");
    
    fmt::println(
        fileout,
        "{},{:.4f},{},{},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f},{:.6f},{:.6f},{:.2f}{}",
        size, temp, run_tag, (is_thermalized ? 1 : 0),
        res_M_abs.mean, res_M_abs.error, res_M_abs.tau_int,
        res_M2.mean, res_M2.error, res_M2.tau_int,
        res_M4.mean, res_M4.error, res_M4.tau_int,
        res_E.mean, res_E.error, res_E.tau_int,
        res_E2.mean, res_E2.error, res_E2.tau_int,
        seed
    );
    fileout.close();
    vprint("Data saved to {}.", filename);
    /* cleanup */
    delete[] magnetization;
    delete[] energy;
    cudaFree(d.lattice);
    cudaFree(d.state);
    cudaFree(d.magnetization);
    cudaFree(d.energy);

    return 0;
}