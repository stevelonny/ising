/* Comprehensive benchmark suite comparing global vs shared memory Ising implementations */
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
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
    cudaMalloc(&d_state, grid.x * block.x * block.y * sizeof(curandStatePhilox4_32_10_t));
    initCurand<<<(grid.x * block.x * block.y + 255) / 256, 256>>>(d_state, 42);

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

int main(int argc, char** argv){
    if(argc < 2) {
        cerr << "Usage: " << argv[0] << " <output_file> [verbose]" << endl;
        return -1;
    }

    string output_file = argv[1];
    bool verbose = (argc > 2 && string(argv[2]) == "verbose");

    vector<int> lattice_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_sweeps = 512;
    double temperature = 2.269; // critical temperature
    
    // Set up exponential table for temperature
    int deltaE[5] = {-8, -4, 0, 4, 8};
    float expTable[5];
    for(int i{0}; i<5; i++){
        expTable[i] = exp(-(float)deltaE[i] / temperature);
    }
    expTable[2] = 1.0f;
    cudaMemcpyToSymbol(d_expTable, expTable, 5 * sizeof(float));

    ofstream outFile(output_file);
    outFile << "implementation,magnetization,size,sweeps,time_ms,throughput_updates_us,time_per_sweep_us" << endl;

    for(int size : lattice_sizes) {
        if(verbose) cout << "Benchmarking size " << size << "..." << endl;
        
        int Len = size * size;
        long long total_updates = (long long)Len * num_sweeps;

        // ====================================================================
        // GPU GLOBAL MEMORY - WITHOUT MAG
        // ====================================================================
        {
            if(verbose) cout << "  GPU (global) - no magnetization..." << endl;
            
            int subLen = size * (size/2);
            int nthreads = subLen;
            if(subLen > 1024) nthreads = 1024;
            int nblocks = (subLen + nthreads - 1) / nthreads;

            int* d_lattice;
            cudaMalloc(&d_lattice, Len * sizeof(int));
            int* d_whites = d_lattice;
            int* d_blacks = d_lattice + subLen;

            curandStatePhilox4_32_10_t* d_state;
            cudaMalloc(&d_state, nblocks * nthreads * sizeof(curandStatePhilox4_32_10_t));
            initCurand<<<nblocks, nthreads>>>(d_state, 42);

            initSublatticeKernel<<<nblocks, nthreads>>>(d_whites, subLen, d_state);
            initSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, subLen, d_state);
            cudaDeviceSynchronize();

            DataBlock d;
            d.lattice = d_lattice;
            d.whites = d_whites;
            d.blacks = d_blacks;
            d.size = size;
            d.Len = Len;
            d.subLen = subLen;
            d.subSize = size/2;
            d.temp = temperature;
            d.state = d_state;
            d.magnetization = nullptr;

            auto start = chrono::high_resolution_clock::now();
            blockSweepKernelGPUNoMag<Periodic>(&d, num_sweeps, nblocks, nthreads);
            cudaDeviceSynchronize();
            auto end = chrono::high_resolution_clock::now();
            auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start);

            double time_ms = duration_us.count() / 1000.0;
            double throughput = total_updates / (double)duration_us.count();
            double time_per_sweep = duration_us.count() / (double)num_sweeps;

            outFile << "GPU_Global,No," << size << "," << num_sweeps << ","
                    << fixed << setprecision(3) << time_ms << ","
                    << throughput << "," << time_per_sweep << endl;

            cudaFree(d_lattice);
            cudaFree(d_state);
        }

        // ====================================================================
        // GPU GLOBAL MEMORY - WITH MAG
        // ====================================================================
        {
            if(verbose) cout << "  GPU (global) - with magnetization..." << endl;
            
            int subLen = size * (size/2);
            int nthreads = subLen;
            if(subLen > 1024) nthreads = 1024;
            int nblocks = (subLen + nthreads - 1) / nthreads;

            int* d_lattice;
            cudaMalloc(&d_lattice, Len * sizeof(int));
            int* d_whites = d_lattice;
            int* d_blacks = d_lattice + subLen;

            curandStatePhilox4_32_10_t* d_state;
            cudaMalloc(&d_state, nblocks * nthreads * sizeof(curandStatePhilox4_32_10_t));
            initCurand<<<nblocks, nthreads>>>(d_state, 42);

            initSublatticeKernel<<<nblocks, nthreads>>>(d_whites, subLen, d_state);
            initSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, subLen, d_state);

            int* d_magnetization;
            cudaMalloc(&d_magnetization, num_sweeps * sizeof(int));
            cudaMemset(d_magnetization, 0, num_sweeps * sizeof(int));

            cudaDeviceSynchronize();

            DataBlock d;
            d.lattice = d_lattice;
            d.whites = d_whites;
            d.blacks = d_blacks;
            d.size = size;
            d.Len = Len;
            d.subLen = subLen;
            d.subSize = size/2;
            d.temp = temperature;
            d.state = d_state;
            d.magnetization = d_magnetization;

            auto start = chrono::high_resolution_clock::now();
            blockSweepKernelGPUWithMag<Periodic>(&d, num_sweeps, nblocks, nthreads);
            cudaDeviceSynchronize();
            auto end = chrono::high_resolution_clock::now();
            auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start);

            double time_ms = duration_us.count() / 1000.0;
            double throughput = total_updates / (double)duration_us.count();
            double time_per_sweep = duration_us.count() / (double)num_sweeps;

            outFile << "GPU_Global,Yes," << size << "," << num_sweeps << ","
                    << fixed << setprecision(3) << time_ms << ","
                    << throughput << "," << time_per_sweep << endl;

            cudaFree(d_lattice);
            cudaFree(d_state);
            cudaFree(d_magnetization);
        }

        // ====================================================================
        // SHARED MEMORY - WITHOUT MAG
        // ====================================================================
        {
            if(verbose) cout << "  Shared memory - no magnetization..." << endl;
            
            int* d_lattice;
            cudaMalloc(&d_lattice, Len * sizeof(int));

            curandStatePhilox4_32_10_t* d_state_init;
            cudaMalloc(&d_state_init, 1024 * sizeof(curandStatePhilox4_32_10_t));
            initCurand<<<4, 256>>>(d_state_init, 42);
            
            initSublatticeKernel<<<(Len + 255) / 256, 256>>>(d_lattice, Len, d_state_init);
            cudaDeviceSynchronize();
            cudaFree(d_state_init);

            auto start = chrono::high_resolution_clock::now();
            blockSweepKernelSharedNoMag<Periodic>(d_lattice, size, num_sweeps);
            cudaDeviceSynchronize();
            auto end = chrono::high_resolution_clock::now();
            auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start);

            double time_ms = duration_us.count() / 1000.0;
            double throughput = total_updates / (double)duration_us.count();
            double time_per_sweep = duration_us.count() / (double)num_sweeps;

            outFile << "Shared_Memory,No," << size << "," << num_sweeps << ","
                    << fixed << setprecision(3) << time_ms << ","
                    << throughput << "," << time_per_sweep << endl;

            cudaFree(d_lattice);
        }

        // ====================================================================
        // SHARED MEMORY - WITH MAG
        // ====================================================================
        {
            if(verbose) cout << "  Shared memory - with magnetization..." << endl;
            
            int* d_lattice;
            cudaMalloc(&d_lattice, Len * sizeof(int));

            curandStatePhilox4_32_10_t* d_state_init;
            cudaMalloc(&d_state_init, 1024 * sizeof(curandStatePhilox4_32_10_t));
            initCurand<<<4, 256>>>(d_state_init, 42);
            
            initSublatticeKernel<<<(Len + 255) / 256, 256>>>(d_lattice, Len, d_state_init);
            cudaDeviceSynchronize();
            cudaFree(d_state_init);

            curandStatePhilox4_32_10_t* d_state;
            int n_states = 256;
            cudaMalloc(&d_state, n_states * sizeof(curandStatePhilox4_32_10_t));
            initCurand<<<1, 256>>>(d_state, 42);

            int* d_magnetization;
            cudaMalloc(&d_magnetization, num_sweeps * sizeof(int));
            cudaMemset(d_magnetization, 0, num_sweeps * sizeof(int));

            cudaDeviceSynchronize();

            DataBlockShared d;
            d.lattice = d_lattice;
            d.size = size;
            d.Len = Len;
            d.state = d_state;
            d.magnetization = d_magnetization;

            auto start = chrono::high_resolution_clock::now();
            blockSweepKernelSharedWithMag<Periodic>(&d, num_sweeps);
            cudaDeviceSynchronize();
            auto end = chrono::high_resolution_clock::now();
            auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start);

            double time_ms = duration_us.count() / 1000.0;
            double throughput = total_updates / (double)duration_us.count();
            double time_per_sweep = duration_us.count() / (double)num_sweeps;

            outFile << "Shared_Memory,Yes," << size << "," << num_sweeps << ","
                    << fixed << setprecision(3) << time_ms << ","
                    << throughput << "," << time_per_sweep << endl;

            cudaFree(d_lattice);
            cudaFree(d_state);
            cudaFree(d_magnetization);
        }
    }

    outFile.close();
    if(verbose) cout << "Benchmark complete. Results written to " << output_file << endl;

    return 0;
}
