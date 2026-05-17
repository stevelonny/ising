#ifndef __ISING_SHARED_HPP__
#define __ISING_SHARED_HPP__

#include <random>
#include <cmath>
#include <curand_kernel.h>

using namespace std;

#ifndef __CONSTANT_D_EXPTABLE__
#define __CONSTANT_D_EXPTABLE__
__constant__ float d_expTable[5];
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
};

#ifndef __INITIALIZE_KERNELS__
#define __INITIALIZE_KERNELS__
__global__ void initCurand(curandStatePhilox4_32_10_t *state, unsigned long seed){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void initSublatticeKernel(int* lattice, int len, curandStatePhilox4_32_10_t *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= len) return;
    float r = curand_uniform(&state[id]);
    lattice[id] = (r < 0.5f) ? -1 : 1;
}
#endif

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
    int exp_idx_w = ((2*spin_w*sum_w) + 8) / 4;
    
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

#endif // __ISING_SHARED_HPP__

