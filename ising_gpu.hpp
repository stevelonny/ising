#include <random>
#include <cmath>
#include <curand_kernel.h>

using namespace std;

struct DataBlock{
    int* whites;
    int* blacks;
    int size;
    int subLen;
    int subSize;
    double temp;
    curandState *state;
    int* magnetization;
};

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
    //left and right index switch places for each change of row's parity

    int sum{0};
    //behold the horror of ternary operators, which however lead to branchless code and avoids warp divergence
    sum += (up >= 0) ? opponent[up] : 0;
    sum += (down < size*size/2) ? opponent[down] : 0;
    sum += opponent[index];
    sum += rowParity ? ((col > 0) ? opponent[index - 1] : 0) : ((col < subSize - 1) ? opponent[index + 1] : 0);

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

__global__ void reduceMagnetizationKernel(const int* whites, const int* blacks, int sublen, int* out){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int partial = 0;
    for (int i = id; i < sublen; i += stride) {
        partial += whites[i];
        partial += blacks[i];
    }

    atomicAdd(out, partial);
}


void blockSweepKernel(DataBlock* d, int blockSweeps, int n_blocks, int n_threads){
    for(int s = 0; s < blockSweeps; s++){
        updateSublatticeKernel<<<n_blocks, n_threads>>>(
            d->whites, d->blacks, true, d->size, d->subSize, d->temp, d->state);
        updateSublatticeKernel<<<n_blocks, n_threads>>>(
            d->blacks, d->whites, false, d->size, d->subSize, d->temp, d->state);
        reduceMagnetizationKernel<<<n_blocks, n_threads>>>(
            d->whites, d->blacks, d->subLen, &d->magnetization[s]);
    }
}

int* initSubLattice(int subLen, std::mt19937& gen){
    int* subLattice = new int[subLen];
    std::bernoulli_distribution dist(0.5);
    for (int i = 0; i < subLen; i++) {
        //subLattice[i] = 1;
        subLattice[i] = dist(gen) ? 1 : -1;
    }
    return subLattice;
}