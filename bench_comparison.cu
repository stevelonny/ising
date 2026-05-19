/* Comprehensive benchmark suite comparing global vs shared memory Ising implementations */
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <random>
#include <iomanip>
#include "ising_gpu.hpp"
#include "ising_shared.hpp"

using namespace std;

// ============================================================================
// GLOBAL MEMORY IMPLEMENTATION VARIANTS
// ============================================================================

template<BoundaryCondition BC>
void blockSweepKernelGPUNoMag(DataBlock* d, int blockSweeps, int n_blocks, int n_threads){
    for(int s = 0; s < blockSweeps; s++){
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->whites, d->blacks, true, d->size, d->subSize, d->state);
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->blacks, d->whites, false, d->size, d->subSize, d->state);
    }
}

template<BoundaryCondition BC>
void blockSweepKernelGPUWithMag(DataBlock* d, int blockSweeps, int n_blocks, int n_threads){
    for(int s = 0; s < blockSweeps; s++){
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->whites, d->blacks, true, d->size, d->subSize, d->state);
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->blacks, d->whites, false, d->size, d->subSize, d->state);
        reduceMagnetizationThrust(d->lattice, d->Len, &d->magnetization[s]);
    }
}

// ============================================================================
// SHARED MEMORY IMPLEMENTATION VARIANTS
// ============================================================================

/* Shared memory without magnetization computation */
template<BoundaryCondition BC>
__global__ void updateTileKernelNoMag(int* lattice, int size, int color, int n_TilesRow, curandStatePhilox4_32_10_t *state){
    /* indexing */
    int tile_id = blockIdx.x;
    int n_coloredTilesRow = n_TilesRow/2; 
    int tile_y = tile_id / (n_coloredTilesRow);
    int tile_rowHalf = tile_id % (n_coloredTilesRow);
    int tile_x = (tile_rowHalf * 2) + ((tile_y & 1) ^ color);
    int tileWidth = blockDim.x * 2;
    int tileHeight = blockDim.y;
    int globalXOrig = tile_x * tileWidth;
    int globalYOrig = tile_y * tileHeight;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;

    extern __shared__ int s_tile_raw[];
    int stride = tileWidth + 2;

    /* read from global memory */
    int total_elements = (tileWidth+2) * (tileHeight+2);
    for(int i{tid}; i<total_elements; i+=total_threads){
        int sy = i / stride;
        int sx = i % stride;
        int gx = globalXOrig + sx - 1;
        int gy = globalYOrig + sy - 1;
        int val = 0;
        if(BC == Open){
            if(gx >= 0 && gx < size && gy >= 0 && gy < size){
                val = lattice[gy * size + gx];
            }
        }
        else if(BC == Periodic){
            int wrapped_x = (gx + size) & (size - 1);
            int wrapped_y = (gy + size) & (size - 1);
            val = lattice[wrapped_y * size + wrapped_x];
        }
        s_tile_raw[sy * stride + sx] = val;
    }
    __syncthreads();

    int gy = globalYOrig + ty;
    int stateIdx = gy * (size/2) + (globalXOrig + 2*tx + (gy&1)) / 2;
    curandStatePhilox4_32_10_t threadState = state[stateIdx];

    /* update logic - whites */
    int sy = ty + 1;
    int sx_w = 2*tx + (gy&1) + 1;
    int sum_w = s_tile_raw[(sy-1)*stride + sx_w] +
                s_tile_raw[(sy+1)*stride + sx_w] +
                s_tile_raw[sy*stride + sx_w-1] +
                s_tile_raw[sy*stride + sx_w+1];
    int spin_w = s_tile_raw[sy*stride + sx_w];
    int exp_idx_w = ((2*spin_w*sum_w) + 8) / 4;
    
    if(curand_uniform(&threadState) < d_expTable[exp_idx_w]){
        spin_w *= -1;
        s_tile_raw[sy*stride + sx_w] = spin_w;
    }
    __syncthreads();

    /* update logic - blacks */
    int sx_b = 2*tx + (1 - (gy&1)) + 1;
    int sum_b = s_tile_raw[(sy-1)*stride + sx_b] +
                s_tile_raw[(sy+1)*stride + sx_b] +
                s_tile_raw[sy*stride + sx_b-1] +
                s_tile_raw[sy*stride + sx_b+1];
    int spin_b = s_tile_raw[sy*stride + sx_b];
    int exp_idx_b = ((2*spin_b*sum_b) + 8) / 4;
    
    if(curand_uniform(&threadState) < d_expTable[exp_idx_b]){
        spin_b *= -1;
        s_tile_raw[sy*stride + sx_b] = spin_b;
    }
    __syncthreads();

    state[stateIdx] = threadState;
    __syncthreads();

    /* write back to global memory */
    int interior_elements = tileWidth * tileHeight;
    for (int i = tid; i < interior_elements; i += total_threads) {
        int local_y = i / tileWidth;
        int local_x = i % tileWidth;
        int write_g_y = globalYOrig + local_y;
        int write_g_x = globalXOrig + local_x;
        lattice[write_g_y * size + write_g_x] = s_tile_raw[(local_y + 1) * stride + (local_x + 1)];
    }
    __syncthreads();
}

template<BoundaryCondition BC>
void blockSweepKernelSharedNoMag(int* d_lattice, int size, int blockSweeps) {
    const int TILE_DIM_X = 16; 
    const int TILE_DIM_Y = 16; 
    
    dim3 block(TILE_DIM_X/2, TILE_DIM_Y);     
    int n_TilesRow = size/TILE_DIM_X;    
    int total_tiles = n_TilesRow * (size/TILE_DIM_Y);    
    dim3 grid(total_tiles/2);
    int tileWidth = TILE_DIM_X + 2;
    int tileHeight = TILE_DIM_Y + 2;
    size_t shared_mem_bytes = (tileWidth*tileHeight) * sizeof(int);

    curandStatePhilox4_32_10_t *d_state;
    int tileStateCount = size * (size/2);
    cudaMalloc(&d_state, tileStateCount * sizeof(curandStatePhilox4_32_10_t));
    initCurand<<<(tileStateCount + 255) / 256, 256>>>(d_state, 42);

    for(int s{0}; s<blockSweeps; s++){
        /* red pass */
        updateTileKernelNoMag<BC><<<grid, block, shared_mem_bytes>>>(
            d_lattice, size, 0, n_TilesRow, d_state
        );
        /* blue pass */        
        updateTileKernelNoMag<BC><<<grid, block, shared_mem_bytes>>>(
            d_lattice, size, 1, n_TilesRow, d_state
        );
    }

    cudaFree(d_state);
}

template<BoundaryCondition BC>
void blockSweepKernelSharedWithMag(DataBlockShared* d, int blockSweeps) {
    const int TILE_DIM_X = 16; 
    const int TILE_DIM_Y = 16; 
    
    dim3 block(TILE_DIM_X/2, TILE_DIM_Y);     
    int n_TilesRow = d->size/TILE_DIM_X;    
    int total_tiles = n_TilesRow * (d->size/TILE_DIM_Y);    
    dim3 grid(total_tiles/2);
    int tileWidth = TILE_DIM_X + 2;
    int tileHeight = TILE_DIM_Y + 2;
    size_t shared_mem_bytes = (tileWidth*tileHeight) * sizeof(int);

    for(int s{0}; s<blockSweeps; s++){
        /* red pass */
        updateTileKernel<BC><<<grid, block, shared_mem_bytes>>>(
            d->lattice, d->size, 0, n_TilesRow, &d->magnetization[s], d->state
        );
        /* blue pass */        
        updateTileKernel<BC><<<grid, block, shared_mem_bytes>>>(
            d->lattice, d->size, 1, n_TilesRow, &d->magnetization[s], d->state
        );
    }
}

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================

template<BoundaryCondition BC>
void runGlobalBenchmark(int size, int sweeps, int repeats, bool withMag, bool verbose){
    const long long total_updates = (long long)size * (long long)size * (long long)sweeps;
    const int subLen = size * (size / 2);
    const int nthreads = (subLen > 1024) ? 1024 : subLen;
    const int nblocks = (subLen + nthreads - 1) / nthreads;

    for(int repeat = 0; repeat < repeats; ++repeat){
        if(verbose){
            cerr << "  GPU (global) repeat " << repeat << (withMag ? " with mag" : " no mag") << endl;
        }

        int* d_lattice = nullptr;
        checkCuda(cudaMalloc(&d_lattice, (long long)size * size * sizeof(int)), "cudaMalloc d_lattice failed");
        int* d_whites = d_lattice;
        int* d_blacks = d_lattice + subLen;

        curandStatePhilox4_32_10_t* d_state = nullptr;
        checkCuda(cudaMalloc(&d_state, subLen * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_state failed");
        initCurand<<<nblocks, nthreads>>>(d_state, 42 + repeat);
        checkCuda(cudaGetLastError(), "initCurand launch failed");

        initSublatticeKernel<<<nblocks, nthreads>>>(d_whites, subLen, d_state);
        checkCuda(cudaGetLastError(), "initSublatticeKernel whites launch failed");
        initSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, subLen, d_state);
        checkCuda(cudaGetLastError(), "initSublatticeKernel blacks launch failed");
        checkCuda(cudaDeviceSynchronize(), "init lattice execution failed");

        int* d_magnetization = nullptr;
        DataBlock d;
        d.lattice = d_lattice;
        d.whites = d_whites;
        d.blacks = d_blacks;
        d.size = size;
        d.Len = size * size;
        d.subLen = subLen;
        d.subSize = size / 2;
        d.temp = 2.269;
        d.state = d_state;
        d.magnetization = nullptr;

        if(withMag){
            checkCuda(cudaMalloc(&d_magnetization, sweeps * sizeof(int)), "cudaMalloc d_magnetization failed");
            checkCuda(cudaMemset(d_magnetization, 0, sweeps * sizeof(int)), "cudaMemset d_magnetization failed");
            d.magnetization = d_magnetization;
        }

        auto start = chrono::high_resolution_clock::now();
        if(withMag){
            blockSweepKernelGPUWithMag<BC>(&d, sweeps, nblocks, nthreads);
        } else {
            blockSweepKernelGPUNoMag<BC>(&d, sweeps, nblocks, nthreads);
        }
        checkCuda(cudaGetLastError(), "global benchmark kernel launch failed");
        checkCuda(cudaDeviceSynchronize(), "global benchmark execution failed");
        auto end = chrono::high_resolution_clock::now();

        auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration_us.count() / 1000.0;
        double throughput = total_updates / (double)duration_us.count();
        double time_per_sweep = duration_us.count() / (double)sweeps;

        cout << "GPU_Global," << (withMag ? "Yes" : "No") << "," << size << "," << sweeps << "," << repeat << ","
             << fixed << setprecision(3) << time_ms << "," << throughput << "," << time_per_sweep << "," << "Periodic" << '\n';

        cudaFree(d_lattice);
        cudaFree(d_state);
        if(d_magnetization){ cudaFree(d_magnetization); }
    }
}

template<BoundaryCondition BC>
void runSharedBenchmark(int size, int sweeps, int repeats, bool withMag, bool cooperative, bool verbose){
    const int TILE_DIM_X = 16;
    const int TILE_DIM_Y = 16;
    const int Len = size * size;
    const int subLen = size * (size / 2);
    const long long total_updates = (long long)Len * (long long)sweeps;
    const int tileStateCount = Len / 2;
    const int nthreads = (tileStateCount > 256) ? 256 : tileStateCount;
    const int nblocks = (tileStateCount + nthreads - 1) / nthreads;

    for(int repeat = 0; repeat < repeats; ++repeat){
        if(verbose){
            cerr << "  Shared " << (cooperative ? "(cooperative)" : "(tile)")
                 << " repeat " << repeat << (withMag ? " with mag" : " no mag") << endl;
        }

        int* d_lattice = nullptr;
        checkCuda(cudaMalloc(&d_lattice, Len * sizeof(int)), "cudaMalloc d_lattice failed");

        curandStatePhilox4_32_10_t* d_state_init = nullptr;
        checkCuda(cudaMalloc(&d_state_init, tileStateCount * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_state_init failed");
        initCurand<<<nblocks, nthreads>>>(d_state_init, 42 + repeat);
        checkCuda(cudaGetLastError(), "initCurand for lattice failed");
        checkCuda(cudaDeviceSynchronize(), "init lattice state execution failed");

        initLatticeKernel<<<nblocks, nthreads>>>(d_lattice, Len, d_state_init);
        checkCuda(cudaGetLastError(), "initLatticeKernel failed");
        checkCuda(cudaDeviceSynchronize(), "lattice init execution failed");
        cudaFree(d_state_init);

        int tileWidth = TILE_DIM_X + 2;
        int tileHeight = TILE_DIM_Y + 2;
        size_t shared_mem_bytes = (tileWidth * tileHeight) * sizeof(int);
        dim3 block(TILE_DIM_X/2, TILE_DIM_Y);
        int n_TilesRow = size / TILE_DIM_X;
        int total_tiles = n_TilesRow * (size / TILE_DIM_Y);
        int totalTilesPerColor = total_tiles / 2;
        dim3 grid(totalTilesPerColor);

        curandStatePhilox4_32_10_t* d_state = nullptr;
        if(withMag || cooperative){
            checkCuda(cudaMalloc(&d_state, tileStateCount * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_state failed");
            initCurand<<<nblocks, nthreads>>>(d_state, 4242 + repeat);
            checkCuda(cudaGetLastError(), "initCurand for shared failed");
            checkCuda(cudaDeviceSynchronize(), "shared state init execution failed");
        }

        int* d_magnetization = nullptr;
        if(withMag){
            checkCuda(cudaMalloc(&d_magnetization, sweeps * sizeof(int)), "cudaMalloc d_magnetization failed");
            checkCuda(cudaMemset(d_magnetization, 0, sweeps * sizeof(int)), "cudaMemset d_magnetization failed");
        }

        auto start = chrono::high_resolution_clock::now();
        if(cooperative){
            DataBlockShared d;
            d.lattice = d_lattice;
            d.size = size;
            d.Len = Len;
            d.subLen = subLen;
            d.subSize = size / 2;
            d.temp = 2.269;
            d.state = d_state;
            d.magnetization = d_magnetization;
            launchCooperativeUpdateTileKernel<BC>(&d, sweeps);
        } else if(withMag){
            DataBlockShared d;
            d.lattice = d_lattice;
            d.size = size;
            d.Len = Len;
            d.subLen = subLen;
            d.subSize = size / 2;
            d.temp = 2.269;
            d.state = d_state;
            d.magnetization = d_magnetization;
            blockSweepKernelSharedWithMag<BC>(&d, sweeps);
        } else {
            blockSweepKernelSharedNoMag<BC>(d_lattice, size, sweeps);
        }
        checkCuda(cudaGetLastError(), "shared benchmark launch failed");
        checkCuda(cudaDeviceSynchronize(), "shared benchmark execution failed");
        auto end = chrono::high_resolution_clock::now();

        auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration_us.count() / 1000.0;
        double throughput = total_updates / (double)duration_us.count();
        double time_per_sweep = duration_us.count() / (double)sweeps;

        cout << (cooperative ? "shared_cooperative" : "Shared_Memory") << "," << (withMag ? "Yes" : "No")
             << "," << size << "," << sweeps << "," << repeat << "," << fixed << setprecision(3) << time_ms
             << "," << throughput << "," << time_per_sweep << "," << "Periodic" << '\n';

        cudaFree(d_lattice);
        if(d_state){ cudaFree(d_state); }
        if(d_magnetization){ cudaFree(d_magnetization); }
    }
}

int main(int argc, char** argv){
    vector<string> args(argv + 1, argv + argc);
    auto require_value = [&](const string& key) -> string {
        for(size_t i = 0; i + 1 < args.size(); ++i){
            if(args[i] == key){
                return args[i + 1];
            }
        }
        throw runtime_error("Missing required argument: " + key);
    };

    if(args.empty() || find(args.begin(), args.end(), "--help") != args.end()){
        cerr << "Usage: " << argv[0]
             << " --impl <global|shared|shared_cooperative> --bc <Periodic|Open> --size <n> --sweeps <n> --mag <0|1> --repeats <n> [--verbose]" << endl;
        return args.empty() ? -1 : 0;
    }

    string impl = require_value("--impl");
    string bc = require_value("--bc");
    int size = stoi(require_value("--size"));
    int sweeps = stoi(require_value("--sweeps"));
    bool mag = stoi(require_value("--mag")) != 0;
    int repeats = stoi(require_value("--repeats"));
    bool verbose = find(args.begin(), args.end(), "--verbose") != args.end();

    if(size <= 0 || sweeps <= 0 || repeats <= 0){
        cerr << "size, sweeps, and repeats must be positive" << endl;
        return -1;
    }
    if(impl == "shared_cooperative" && !mag){
        cerr << "shared_cooperative requires --mag 1" << endl;
        return -1;
    }

    int deltaE[5] = {-8, -4, 0, 4, 8};
    float expTable[5];
    double temperature = 2.269;
    for(int i = 0; i < 5; ++i){
        expTable[i] = exp(-(float)deltaE[i] / temperature);
    }
    expTable[2] = 1.0f;
    checkCuda(cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float)), "copy expTable failed");

    cout << "implementation,magnetization,size,sweeps,repeat,time_ms,throughput_updates_us,time_per_sweep_us,bc" << '\n';

    if(bc == "Periodic" || bc == "pbc"){
        if(impl == "global"){
            runGlobalBenchmark<Periodic>(size, sweeps, repeats, mag, verbose);
        } else if(impl == "shared"){
            runSharedBenchmark<Periodic>(size, sweeps, repeats, mag, false, verbose);
        } else if(impl == "shared_cooperative"){
            runSharedBenchmark<Periodic>(size, sweeps, repeats, true, true, verbose);
        } else {
            cerr << "Unknown impl: " << impl << endl;
            return -1;
        }
    } else if(bc == "Open" || bc == "obc"){
        if(impl == "global"){
            runGlobalBenchmark<Open>(size, sweeps, repeats, mag, verbose);
        } else if(impl == "shared"){
            runSharedBenchmark<Open>(size, sweeps, repeats, mag, false, verbose);
        } else if(impl == "shared_cooperative"){
            runSharedBenchmark<Open>(size, sweeps, repeats, true, true, verbose);
        } else {
            cerr << "Unknown impl: " << impl << endl;
            return -1;
        }
    } else {
        cerr << "Unknown boundary condition: " << bc << endl;
        return -1;
    }

    return 0;
}
