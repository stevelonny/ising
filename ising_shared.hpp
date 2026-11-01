#ifndef __ISING_SHARED_HPP__
#define __ISING_SHARED_HPP__

#include <random>
#include <cmath>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;
using namespace std;

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(-1);
    }
}

#define __TILE_DIM_X__ 32
#define __TILE_DIM_Y__ 32

#ifndef __CONSTANT_D_EXPTABLE__
#define __CONSTANT_D_EXPTABLE__
__constant__ float d_expTable[9];
const int _deltaE[9] = {-8, -6, -4, -2, 0, 2, 4, 6, 8};
#endif

#ifndef __BOUNDARY_CONDITION__
#define __BOUNDARY_CONDITION__
enum BoundaryCondition{
    Open,
    Periodic
};
#endif

struct DataBlockShared{//all pointers are device pointers
    int* lattice;
    int size;
    int Len;
    int subLen;
    int subSize;
    double temp;
    curandStatePhilox4_32_10_t *state;
    int* magnetization;
    int* energy;
};

struct KernelConfig {
    dim3 block;
    dim3 grid;
    size_t shared_mem_bytes;
    int n_TilesRow;
};

#ifndef __INITIALIZE_CURAND__
#define __INITIALIZE_CURAND__
__global__ void initCurand(curandStatePhilox4_32_10_t *state, unsigned long seed){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}
#endif

//random lattice initialization kernel
__global__ void initLatticeKernel(int* lattice, int len, curandStatePhilox4_32_10_t *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= len / 2) return; //only subLen threads run this
    
    curandStatePhilox4_32_10_t localState = state[id];
    
    //initialize two spins per state to cover the whole lattice
    lattice[id*2]     = (curand_uniform(&localState) < 0.5f) ? -1 : 1;
    lattice[id*2 + 1] = (curand_uniform(&localState) < 0.5f) ? -1 : 1;
    
    state[id] = localState;
}

//cold(+1) lattice initialization kernel
__global__ void initColdLatticeKernel(int* lattice, int len){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= len) return;
    lattice[id] = 1; //all spins up
}

//helper function to configure kernel launch parameters in shared approach
KernelConfig kernelConfig(int size)
{
    const int TILE_DIM_X = __TILE_DIM_X__;
    const int TILE_DIM_Y = __TILE_DIM_Y__;
    KernelConfig config;

    config.n_TilesRow = size / TILE_DIM_X;

    dim3 block(TILE_DIM_X / 2, TILE_DIM_Y);
    int n_TilesRow = size / TILE_DIM_X; //number of tiles per row (each tile is blockDimX*2 pixels wide or TILE_DIM_X pixels wide)
    int total_tiles = n_TilesRow * (size / TILE_DIM_Y);
    dim3 grid(total_tiles / 2);
    int tileWidth = TILE_DIM_X + 2;     //tile size in pixels (halo added in kernel)
    int tileHeight = TILE_DIM_Y + 2;    //tile size in pixels (halo added in kernel)
    size_t shared_mem_bytes = (tileWidth * tileHeight) * sizeof(int);
    config.block = block;
    config.grid = grid;
    config.shared_mem_bytes = shared_mem_bytes;
    return config;
}

//kernel updates the lattice without reduction
template<BoundaryCondition BC>
__global__ void updateTileKernel(int* lattice, int size, int color, int n_TilesRow, curandStatePhilox4_32_10_t *state){
    
    /* indexing */
    //tiling
    int tile_id = blockIdx.x;
    int n_coloredTilesRow = n_TilesRow/2; 
    int tile_y = tile_id / (n_coloredTilesRow);
    int tile_rowHalf = tile_id % (n_coloredTilesRow);
    int tile_x = (tile_rowHalf * 2) + ((tile_y & 1) ^ color);
    int tileWidth = blockDim.x * 2; //tile size = 2*threads to accomodate checkerboard pattern
    int tileHeight = blockDim.y;
    //global indexing of tile
    int globalXOrig = tile_x * tileWidth;
    int globalYOrig = tile_y * tileHeight;
    //local indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y; //threads per block

    //shared memory (tile + halo)
    //will reuse the same memory for reduction
    //size = (tileWidth+2) * (tileHeight+2)
    extern __shared__ int s_tile_raw[];
    int stride = tileWidth + 2;
    //will need indexing for the tile sx, sy for accessing the halo

    /* read from global memory */
    int total_elements = (tileWidth+2) * (tileHeight+2);
    for(int i{tid}; i<total_elements; i+=total_threads){
        //indexing in the shared memory tile
        int sy = i / stride;
        int sx = i % stride;
        int gx = globalXOrig + sx - 1;
        int gy = globalYOrig + sy - 1;
        //default read = 0 for boundaries conditions
        int val = 0;
        if(BC == Open){
            if(gx >= 0 && gx < size && gy >= 0 && gy < size){
                val = lattice[gy * size + gx];
            }
        }
        else if(BC == Periodic){//penalty
            int wrapped_x = (gx + size) & (size - 1);
            int wrapped_y = (gy + size) & (size - 1);
            val = lattice[wrapped_y * size + wrapped_x];
        }
        s_tile_raw[sy * stride + sx] = val;
    }
    __syncthreads();

    /* interlude */
    int gy = globalYOrig + ty;
    //corresponding white/black use the same state -> size/2
    //globalXOrig + 2*tx + (gy&1) computes columns + alternate parity
    int stateIdx = gy * (size/2) + (globalXOrig + 2*tx + (gy&1)) / 2;
    curandStatePhilox4_32_10_t threadState = state[stateIdx]; //64 bytes

    /* udpate logic*/
    /* whites */
    int sy = ty + 1; //skip halo
    int sx_w = 2*tx + (gy&1) + 1;
    int sum_w = s_tile_raw[(sy-1)*stride + sx_w] +  //up
                s_tile_raw[(sy+1)*stride + sx_w] +  //down
                s_tile_raw[sy*stride + sx_w-1] +    //left
                s_tile_raw[sy*stride + sx_w+1];     //right
    int spin_w = s_tile_raw[sy*stride + sx_w];
    //int exp_idx_w = ((2*spin_w*sum_w) + 8) / 4;
    int exp_idx_w = spin_w*sum_w + 4;

    if(curand_uniform(&threadState) < d_expTable[exp_idx_w]){
        spin_w *= -1;
        s_tile_raw[sy*stride + sx_w] = spin_w;
    }
    __syncthreads();

    /* blacks */
    int sx_b = 2*tx + (1 - (gy&1)) + 1;
    int sum_b = s_tile_raw[(sy-1)*stride + sx_b] +  //up
                s_tile_raw[(sy+1)*stride + sx_b] +  //down
                s_tile_raw[sy*stride + sx_b-1] +    //left
                s_tile_raw[sy*stride + sx_b+1];     //right
    int spin_b = s_tile_raw[sy*stride + sx_b];
    //int exp_idx_b = ((2*spin_b*sum_b) + 8) / 4;
    int exp_idx_b = spin_b*sum_b + 4;
    
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

//kernel updates the lattice with reduction for magnetization
template<BoundaryCondition BC>
__global__ void updateTileKernel(int* lattice, int size, int color, int n_TilesRow, int* d_mag_sweep, curandStatePhilox4_32_10_t *state){
    
    /* indexing */
    //tiling
    int tile_id = blockIdx.x;
    int n_coloredTilesRow = n_TilesRow/2; 
    int tile_y = tile_id / (n_coloredTilesRow);
    int tile_rowHalf = tile_id % (n_coloredTilesRow);
    int tile_x = (tile_rowHalf * 2) + ((tile_y & 1) ^ color);
    int tileWidth = blockDim.x * 2; //tile size = 2*threads to accomodate checkerboard pattern
    int tileHeight = blockDim.y;
    //global indexing of tile
    int globalXOrig = tile_x * tileWidth;
    int globalYOrig = tile_y * tileHeight;
    //local indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y; //threads per block

    //shared memory (tile + halo)
    //will reuse the same memory for reduction
    extern __shared__ int s_tile_raw[];
    int stride = tileWidth + 2;
    //will need indexing for the tile sx, sy for accessing the halo

    /* read from global memory */
    int total_elements = (tileWidth+2) * (tileHeight+2);
    for(int i{tid}; i<total_elements; i+=total_threads){
        //indexing in the shared memory tile
        int sy = i / stride;
        int sx = i % stride;
        int gx = globalXOrig + sx - 1;
        int gy = globalYOrig + sy - 1;
        //default read = 0 for boundaries conditions
        int val = 0;
        if(BC == Open){
            if(gx >= 0 && gx < size && gy >= 0 && gy < size){
                val = lattice[gy * size + gx];
            }
        }
        else if(BC == Periodic){//penalty
            int wrapped_x = (gx + size) & (size - 1);
            int wrapped_y = (gy + size) & (size - 1);
            val = lattice[wrapped_y * size + wrapped_x];
        }
        s_tile_raw[sy * stride + sx] = val;
    }
    __syncthreads();

    /* interlude */
    int gy = globalYOrig + ty;
    //corresponding white/black use the same state -> size/2
    //globalXOrig + 2*tx + (gy&1) computes columns + alternate parity
    int stateIdx = gy * (size/2) + (globalXOrig + 2*tx + (gy&1)) / 2;
    curandStatePhilox4_32_10_t threadState = state[stateIdx]; //64 bytes

    /* udpate logic*/
    /* whites */
    int sy = ty + 1; //skip halo
    int sx_w = 2*tx + (gy&1) + 1;
    int sum_w = s_tile_raw[(sy-1)*stride + sx_w] +  //up
                s_tile_raw[(sy+1)*stride + sx_w] +  //down
                s_tile_raw[sy*stride + sx_w-1] +    //left
                s_tile_raw[sy*stride + sx_w+1];     //right
    int spin_w = s_tile_raw[sy*stride + sx_w];
    //int exp_idx_w = ((2*spin_w*sum_w) + 8) / 4;
    int exp_idx_w = spin_w*sum_w + 4;

    if(curand_uniform(&threadState) < d_expTable[exp_idx_w]){
        spin_w *= -1;
        s_tile_raw[sy*stride + sx_w] = spin_w;
    }
    __syncthreads();

    /* blacks */
    int sx_b = 2*tx + (1 - (gy&1)) + 1;
    int sum_b = s_tile_raw[(sy-1)*stride + sx_b] +  //up
                s_tile_raw[(sy+1)*stride + sx_b] +  //down
                s_tile_raw[sy*stride + sx_b-1] +    //left
                s_tile_raw[sy*stride + sx_b+1];     //right
    int spin_b = s_tile_raw[sy*stride + sx_b];
    //int exp_idx_b = ((2*spin_b*sum_b) + 8) / 4;
    int exp_idx_b = spin_b*sum_b + 4;
    
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
    /* reduction */
    if(color == 1)
    {
        int tMag = spin_w + spin_b;
        int* s_reduce = s_tile_raw;
        s_reduce[tid] = tMag;
        __syncthreads();

        //parallel reduction in shared memory
        for(int s{total_threads/2}; s>0; s/=2){
            if(tid < s){
                s_tile_raw[tid] += s_tile_raw[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0){
            atomicAdd(d_mag_sweep, s_tile_raw[0]);
        }
    }

}

//kernel updates the lattice with reduction for magnetization and energy
template<BoundaryCondition BC>
__global__ void updateTileKernel(
    int* lattice, int size, int color, int n_TilesRow,
    int* d_mag_sweep, int* d_energy_sweep, curandStatePhilox4_32_10_t *state)
{    
    /* indexing */
    //tiling
    int tile_id = blockIdx.x;
    int n_coloredTilesRow = n_TilesRow/2; 
    int tile_y = tile_id / (n_coloredTilesRow);
    int tile_rowHalf = tile_id % (n_coloredTilesRow);
    int tile_x = (tile_rowHalf * 2) + ((tile_y & 1) ^ color);
    int tileWidth = blockDim.x * 2; //tile size = 2*threads to accomodate checkerboard pattern
    int tileHeight = blockDim.y;
    //global indexing of tile
    int globalXOrig = tile_x * tileWidth;
    int globalYOrig = tile_y * tileHeight;
    //local indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y; //threads per block

    //shared memory (tile + halo)
    //will reuse the same memory for reduction
    extern __shared__ int s_tile_raw[];
    int stride = tileWidth + 2;
    //will need indexing for the tile sx, sy for accessing the halo

    /* read from global memory */
    int total_elements = (tileWidth+2) * (tileHeight+2);
    for(int i{tid}; i<total_elements; i+=total_threads){
        //indexing in the shared memory tile
        int sy = i / stride;
        int sx = i % stride;
        int gx = globalXOrig + sx - 1;
        int gy = globalYOrig + sy - 1;
        //default read = 0 for boundaries conditions
        int val = 0;
        if(BC == Open){
            if(gx >= 0 && gx < size && gy >= 0 && gy < size){
                val = lattice[gy * size + gx];
            }
        }
        else if(BC == Periodic){//penalty
            int wrapped_x = (gx + size) & (size - 1);
            int wrapped_y = (gy + size) & (size - 1);
            val = lattice[wrapped_y * size + wrapped_x];
        }
        s_tile_raw[sy * stride + sx] = val;
    }
    __syncthreads();

    /* interlude */
    int gy = globalYOrig + ty;
    //corresponding white/black use the same state -> size/2
    //globalXOrig + 2*tx + (gy&1) computes columns + alternate parity
    int stateIdx = gy * (size/2) + (globalXOrig + 2*tx + (gy&1)) / 2;
    curandStatePhilox4_32_10_t threadState = state[stateIdx]; //64 bytes

    /* udpate logic*/
    /* whites */
    int sy = ty + 1; //skip halo
    int sx_w = 2*tx + (gy&1) + 1;
    int sum_w = s_tile_raw[(sy-1)*stride + sx_w] +  //up
                s_tile_raw[(sy+1)*stride + sx_w] +  //down
                s_tile_raw[sy*stride + sx_w-1] +    //left
                s_tile_raw[sy*stride + sx_w+1];     //right
    int spin_w = s_tile_raw[sy*stride + sx_w];
    //int exp_idx_w = ((2*spin_w*sum_w) + 8) / 4;
    int exp_idx_w = spin_w*sum_w + 4;
    
    if(curand_uniform(&threadState) < d_expTable[exp_idx_w]){
        spin_w *= -1;
        s_tile_raw[sy*stride + sx_w] = spin_w;
    }
    __syncthreads();

    /* blacks */
    int sx_b = 2*tx + (1 - (gy&1)) + 1;
    int sum_b = s_tile_raw[(sy-1)*stride + sx_b] +  //up
                s_tile_raw[(sy+1)*stride + sx_b] +  //down
                s_tile_raw[sy*stride + sx_b-1] +    //left
                s_tile_raw[sy*stride + sx_b+1];     //right
    int spin_b = s_tile_raw[sy*stride + sx_b];
    //int exp_idx_b = ((2*spin_b*sum_b) + 8) / 4;
    int exp_idx_b = spin_b*sum_b + 4;

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
    /* reduction */
    if(color == 1)
    {
        //save into registers nbs for energy reduction
        int nb_up_w     = s_tile_raw[(sy-1)*stride + sx_w];
        int nb_right_w  = s_tile_raw[sy*stride + sx_w+1];
        int nb_up_b     = s_tile_raw[(sy-1)*stride + sx_b];
        int nb_right_b  = s_tile_raw[sy*stride + sx_b+1];
        //magnetization
        int tMag = spin_w + spin_b;
        int* s_reduce = s_tile_raw;
        s_reduce[tid] = tMag;
        __syncthreads();

        //parallel reduction in shared memory
        for(int s{total_threads/2}; s>0; s/=2){
            if(tid < s){
                s_tile_raw[tid] += s_tile_raw[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0){
            atomicAdd(d_mag_sweep, s_tile_raw[0]);
        }
        __syncthreads();
        //energy
        int tEnergy = - spin_w * (nb_up_w+nb_right_w) - spin_b * (nb_up_b+nb_right_b);
        s_reduce[tid] = tEnergy;
        __syncthreads();

        //parallel reduction in shared memory
        for(int s{total_threads/2}; s>0; s/=2){
            if(tid < s){
                s_tile_raw[tid] += s_tile_raw[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0){
            atomicAdd(d_energy_sweep, s_tile_raw[0]);
        }
        __syncthreads();
    }

}

//kernel for cooperative groups
template<BoundaryCondition BC>
__device__ void updateTile(int* lattice, int size, int color, int tile_id, int n_TilesRow, int* d_mag_sweep,  int* s_tile_raw, curandStatePhilox4_32_10_t *state){
    
    /* indexing */
    //tiling
    //int tile_id = blockIdx.x;
    //now tile_id is passed
    int n_coloredTilesRow = n_TilesRow/2; 
    int tile_y = tile_id / (n_coloredTilesRow);
    int tile_rowHalf = tile_id % (n_coloredTilesRow);
    int tile_x = (tile_rowHalf * 2) + ((tile_y & 1) ^ color);
    int tileWidth = blockDim.x * 2; //tile size = 2*threads to accomodate checkerboard pattern
    int tileHeight = blockDim.y;
    //global indexing of tile
    int globalXOrig = tile_x * tileWidth;
    int globalYOrig = tile_y * tileHeight;
    //local indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y; //threads per block

    int stride = tileWidth + 2;
    //will need indexing for the tile sx, sy for accessing the halo

    /* read from global memory */
    int total_elements = (tileWidth+2) * (tileHeight+2);
    for(int i{tid}; i<total_elements; i+=total_threads){
        //indexing in the shared memory tile
        int sy = i / stride;
        int sx = i % stride;
        int gx = globalXOrig + sx - 1;
        int gy = globalYOrig + sy - 1;
        //default read = 0 for boundaries conditions
        int val = 0;
        if(BC == Open){
            if(gx >= 0 && gx < size && gy >= 0 && gy < size){
                val = lattice[gy * size + gx];
            }
        }
        else if(BC == Periodic){//penalty
            int wrapped_x = (gx + size) & (size - 1);
            int wrapped_y = (gy + size) & (size - 1);
            val = lattice[wrapped_y * size + wrapped_x];
        }
        s_tile_raw[sy * stride + sx] = val;
    }
    __syncthreads();

    /* interlude */
    int gy = globalYOrig + ty;
    //corresponding white/black use the same state -> size/2
    //globalXOrig + 2*tx + (gy&1) computes columns + alternate parity
    int stateIdx = gy * (size/2) + (globalXOrig + 2*tx + (gy&1)) / 2;
    curandStatePhilox4_32_10_t threadState = state[stateIdx]; //64 bytes

    /* udpate logic*/
    /* whites */
    int sy = ty + 1; //skip halo
    int sx_w = 2*tx + (gy&1) + 1;
    int sum_w = s_tile_raw[(sy-1)*stride + sx_w] +  //up
                s_tile_raw[(sy+1)*stride + sx_w] +  //down
                s_tile_raw[sy*stride + sx_w-1] +    //left
                s_tile_raw[sy*stride + sx_w+1];     //right
    int spin_w = s_tile_raw[sy*stride + sx_w];
    //int exp_idx_w = ((2*spin_w*sum_w) + 8) / 4; //need to fix
    int exp_idx_w = spin_w*sum_w + 4;
    
    if(curand_uniform(&threadState) < d_expTable[exp_idx_w]){
        spin_w *= -1;
        s_tile_raw[sy*stride + sx_w] = spin_w;
    }
    __syncthreads();

    /* blacks */
    int sx_b = 2*tx + (1 - (gy&1)) + 1;
    int sum_b = s_tile_raw[(sy-1)*stride + sx_b] +  //up
                s_tile_raw[(sy+1)*stride + sx_b] +  //down
                s_tile_raw[sy*stride + sx_b-1] +    //left
                s_tile_raw[sy*stride + sx_b+1];     //right
    int spin_b = s_tile_raw[sy*stride + sx_b];
    //int exp_idx_b = ((2*spin_b*sum_b) + 8) / 4;
    int exp_idx_b = spin_b*sum_b + 4;
    
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
    /* reduction */
    int tMag = spin_w + spin_b;
    int* s_reduce = s_tile_raw;
    s_reduce[tid] = tMag;
    __syncthreads();

    //parallel reduction in shared memory
    for(int s{total_threads/2}; s>0; s/=2){
        if(tid < s){
            s_tile_raw[tid] += s_tile_raw[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd(d_mag_sweep, s_tile_raw[0]);
    }

}

//kernel launcher for cooperative groups
template<BoundaryCondition BC>
__global__ void cooperativeUpdateTileKernel(DataBlockShared d, int blocksweeps, int totalTilesPerColor){
    cg::grid_group grid = cg::this_grid();
    
    const int TILE_DIM_X = __TILE_DIM_X__;
    int n_TilesRow = d.size / TILE_DIM_X; 

    extern __shared__ int s_tile_raw[];

    for(int s{0}; s<blocksweeps; s++){
        //each tile update will need to be strided
        /* red pass */
        for(size_t tile_id{blockIdx.x}; tile_id < totalTilesPerColor; tile_id += gridDim.x){
            updateTile<BC>(d.lattice, d.size, 0, tile_id, n_TilesRow, &d.magnetization[s], s_tile_raw, d.state);
        }
        grid.sync();
        /* blue pass */
        for(size_t tile_id{blockIdx.x}; tile_id < totalTilesPerColor; tile_id += gridDim.x){
            updateTile<BC>(d.lattice, d.size, 1, tile_id, n_TilesRow, &d.magnetization[s], s_tile_raw, d.state);
        }
        grid.sync();
    }

}

//launches cooperative group kernel
template<BoundaryCondition BC>
void launchCooperativeUpdateTileKernel(DataBlockShared* d, int blockSweeps){
    const int TILE_DIM_X = __TILE_DIM_X__;
    const int TILE_DIM_Y = __TILE_DIM_Y__;
    
    dim3 block(TILE_DIM_X/2, TILE_DIM_Y);     
    int n_TilesRow = d->size/TILE_DIM_X;    
    int total_tiles = n_TilesRow * (d->size/TILE_DIM_Y);
    int totalTilesPerColor = total_tiles / 2;
    //dim3 grid(total_tiles/2);
    //dimension of the grid must be adjusted based on maximumn occupancy
    int tileWidth = TILE_DIM_X + 2;
    int tileHeight = TILE_DIM_Y + 2;
    size_t shared_mem_bytes = (tileWidth*tileHeight) * sizeof(int);

    int numBlocksPerStreamingProcessor{0};
    int numStreamingProcessors{0};

    cudaDeviceGetAttribute(&numStreamingProcessors, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerStreamingProcessor,
        (const void*)cooperativeUpdateTileKernel<BC>,
        block.x * block.y * block.z,
        shared_mem_bytes
    );

    int maxBlocks = numBlocksPerStreamingProcessor * numStreamingProcessors;
    //if the lattice is smaller, use less blocks
    //int activeBlocks = std::min(maxBlocks, totalTilesPerColor);
    int activeBlocks = maxBlocks;
    while(activeBlocks > 0){
        if(totalTilesPerColor % activeBlocks == 0){
            break;
        }
        activeBlocks--;
    }

    if(activeBlocks < (numStreamingProcessors*2)){
        activeBlocks = std::min(maxBlocks, totalTilesPerColor);
    }

    dim3 grid(activeBlocks);

    DataBlockShared d_val = *d;

    void* args[] = {
        &d_val,
        &blockSweeps,
        &totalTilesPerColor
    };

    checkCuda(cudaLaunchCooperativeKernel(
        (const void*)cooperativeUpdateTileKernel<BC>,
        grid,
        block,
        args,
        shared_mem_bytes,
        0
    ), "Failed to launch cooperative update tile kernel");

}

//says "int blockSweeps" but it is actually the number of sweeps to perform
//old remains of old blocking implementation
template<BoundaryCondition BC>
void blockSweepKernelShared(DataBlockShared* d, int blockSweeps) {
    KernelConfig config = kernelConfig(d->size);

    for(int s{0}; s<blockSweeps; s++){
        /* red pass */
        updateTileKernel<BC><<<config.grid, config.block, config.shared_mem_bytes>>>(
            d->lattice, 
            d->size, 
            0,
            config.n_TilesRow, 
            &d->magnetization[s],
            &d->energy[s],
            d->state
        );
        /* blue pass */        
        updateTileKernel<BC><<<config.grid, config.block, config.shared_mem_bytes>>>(
            d->lattice, 
            d->size, 
            1,
            config.n_TilesRow, 
            &d->magnetization[s],
            &d->energy[s],
            d->state
        );
    }
}


#endif // __ISING_SHARED_HPP__

