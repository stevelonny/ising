#include <random>
#include <cmath>
#include <curand_kernel.h>

using namespace std;

__global__ void initCurand(curandState *state, unsigned long seed){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void initSublatticeKernel(int* subLattice, int sublen, curandState *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= sublen) return;
    float r = curand_uniform(&state[id]);
    subLattice[id] = (r < 0.5f) ? -1 : 1;
}

__device__ int getNeighbourSum_obc(int* opponent, int index, int size, int subSize, bool isWhite){
    int row = index / subSize;
    int col = index % subSize;
    bool rowParity = (row % 2 == 0);
    if(isWhite) rowParity = !rowParity; //white and black sublattice have opposite parity
    /* nn indices */
    int up      = index - subSize;
    int down    = index + subSize;
    // left and right index switch places for each change of row's parity ("adjacent")

    int sum{0};
    
    if (up >= 0)            sum += opponent[up];
    if (down < size*size/2) sum += opponent[down];
    if (rowParity){ //even row
        sum += opponent[index];                     //right
        if(col > 0) sum += opponent[index - 1];     //left
    }
    else{ //odd row
        sum += opponent[index];                             //left
        if(col < subSize - 1) sum += opponent[index + 1];   //right
    }

    return sum;
}

__global__ void updateSublatticeKernel(int* subLattice, int* opponent, bool isWhite, int size, int subSize, double temp, curandState *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= size * subSize / 2) return;
    int nearSum = getNeighbourSum_obc(opponent, id, size, subSize, isWhite);
    double deltaE = 2.*(double)subLattice[id] * (double)nearSum;
    float r = curand_uniform(&state[id]);
    if(r < exp(-deltaE / temp)){
        subLattice[id] *= -1;
    }
}

__global__ void reduceMagnetizationKernel(const int* sublattice, int sublen, int* out){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int partial = 0;
    for (int i = id; i < sublen; i += stride) {
        partial += sublattice[i];
    }
    atomicAdd(out, partial);
}

/* void blockSweepKernel(int blockSweeps, int n_block, int* whites, int* blacks, int size, int subSize, double temp, int* whiteMag, int* blackMag, int* d_whiteMagTmp, int* d_blackMagTmp, curandState *state){
    int blocks = NBLOCKS;
    int threads = NTHREADS;

    for(int s = 0; s < blockSweeps; s++){
        updateSublatticeKernel<<<blocks, threads>>>(whites, blacks, true, size, subSize, temp, state);
        updateSublatticeKernel<<<blocks, threads>>>(blacks, whites, false, size, subSize, temp, state);
    }

    cudaMemset(d_whiteMagTmp, 0, sizeof(int));
    cudaMemset(d_blackMagTmp, 0, sizeof(int));
    reduceMagnetizationKernel<<<blocks, threads>>>(whites, subSize * size, d_whiteMagTmp);
    reduceMagnetizationKernel<<<blocks, threads>>>(blacks, subSize * size, d_blackMagTmp);

    cudaMemcpy(&whiteMag[n_block], d_whiteMagTmp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&blackMag[n_block], d_blackMagTmp, sizeof(int), cudaMemcpyDeviceToHost);
} */

int* initSubLattice(int subLen, std::mt19937& gen){
    int* subLattice = new int[subLen];
    std::bernoulli_distribution dist(0.5);
    for (int i = 0; i < subLen; i++) {
        //subLattice[i] = 1;
        subLattice[i] = dist(gen) ? 1 : -1;
    }
    return subLattice;
}