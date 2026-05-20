#include "ising_shared.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void dummy_kernel() {}

int main(){
    const int TILE_DIM_X = 16;
    const int TILE_DIM_Y = 16;
    dim3 block(TILE_DIM_X/2, TILE_DIM_Y);
    int threadsPerBlock = block.x * block.y * block.z;
    int tileWidth = TILE_DIM_X + 2;
    int tileHeight = TILE_DIM_Y + 2;
    size_t shared_mem_bytes = (tileWidth * tileHeight) * sizeof(int);

    int numBlocksPerSM = 0;
    cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        (const void*)cooperativeUpdateTileKernel<Periodic>,
        threadsPerBlock,
        shared_mem_bytes
    );
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    std::cout << "threadsPerBlock=" << threadsPerBlock << " shared_mem_bytes=" << shared_mem_bytes << "\n";
    std::cout << "cudaOccupancyMaxActiveBlocksPerMultiprocessor rc=" << rc << " numBlocksPerSM=" << numBlocksPerSM << " numSMs=" << numSMs << "\n";
    int sizes[] = {32,64,128,256,512,1024,2048,4096,8192};
    for(int size : sizes){
        int n_TilesRow = size / TILE_DIM_X;
        int total_tiles = n_TilesRow * (size / TILE_DIM_Y);
        int totalTilesPerColor = total_tiles / 2;
        int maxBlocks = numBlocksPerSM * numSMs;
        //int activeBlocks = std::min(maxBlocks, totalTilesPerColor);
        int activeBlocks = maxBlocks;
        while(activeBlocks > 0){
            if(totalTilesPerColor % activeBlocks == 0){
                break;
            }
            activeBlocks--;
        }
        if(activeBlocks < (numSMs*2)){
            activeBlocks = std::min(maxBlocks, totalTilesPerColor);
        }
        std::cout << "size=" << size << " n_TilesRow="<<n_TilesRow<<" total_tiles="<<total_tiles
                  <<" totalTilesPerColor="<<totalTilesPerColor<<" maxBlocks="<<maxBlocks<<" activeBlocks="<<activeBlocks<<"\n";
    }
    return 0;

}