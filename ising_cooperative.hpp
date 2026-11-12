#include "ising_shared.hpp"

#ifndef __ISING_COOOPERATIVE_HPP__
#define __ISING_COOOPERATIVE_HPP__

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


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

#endif // __ISING_COOPERATIVE_HPP__