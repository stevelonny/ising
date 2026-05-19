#include "ising_shared.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>



int main(){

    int numBlocksPerStreamingProcessor{0};
    int numStreamingProcessors{0};

    cudaDeviceGetAttribute(&numStreamingProcessors, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerStreamingProcessor,
        (const void*)cooperativeUpdateTileKernel<Open>,
        8 * 16,
        18*18*sizeof(int)
    );


    cout << "Number of streaming processors: " << numStreamingProcessors << "\n";
    cout << "Max blocks per streaming processor for cooperative kernel: " << numBlocksPerStreamingProcessor << "\n";

    return 0;

}