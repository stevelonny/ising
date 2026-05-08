#include <random>
#include <cmath>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

using namespace std;

enum BoundaryCondition{
    Open,
    Periodic
    //Fixed
};

struct DataBlock{//all pointers are device pointers
    int* lattice;
    int* whites;
    int* blacks;
    int size;
    int Len;
    int subLen;
    int subSize;
    double temp;
    curandStatePhilox4_32_10_t *state;
    int* magnetization;
    //int* magnetization_thrust;
};

__global__ void initCurand(curandStatePhilox4_32_10_t *state, unsigned long seed){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void initSublatticeKernel(int* subLattice, int sublen, curandStatePhilox4_32_10_t *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= sublen) return;
    float r = curand_uniform(&state[id]);
    subLattice[id] = (r < 0.5f) ? -1 : 1;
}

template<BoundaryCondition BC>
__device__ inline int getNeighbourSum(int* opponent, int index, int size, int subSize, bool isWhite){
    int row = index / subSize;
    int col = index % subSize;
    bool rowParity = (row % 2 == 0);
    if(isWhite) rowParity = !rowParity; //white and black sublattice have opposite parity
    /* nn indices */
    int up      = index - subSize;
    int down    = index + subSize;
    //left and right index switch places for each change of row's parity

    int sum{0};
    if(BC == Open){
        //behold the horror of ternary operators, which however lead to branchless code and avoids warp divergence
        sum += (up >= 0) ? opponent[up] : 0;
        sum += (down < size*subSize) ? opponent[down] : 0;
        sum += opponent[index];
        sum += rowParity ?
               ((col > 0) ? opponent[index - 1] : 0) :
               ((col < subSize - 1) ? opponent[index + 1] : 0);
    }
    else if(BC == Periodic){
        // Vertical periodicity with bitwise AND (assumes power of 2)
        int subLen = size * subSize;
        //int up_wrapped = (index - subSize + subLen) & (subLen - 1);
        //int down_wrapped = (index + subSize) & (subLen - 1);
        
        sum += opponent[(index - subSize + subLen) & (subLen - 1)];
        sum += opponent[(index + subSize) & (subLen - 1)];
        sum += opponent[index];
        //horizontal neighbour
        int col_left = (col - 1 + subSize) & (subSize - 1);
        int col_right = (col + 1) & (subSize - 1);
        int neighbor_col = rowParity ? col_left : col_right;
        sum += opponent[row * subSize + neighbor_col];
    }
    return sum;
}

template<BoundaryCondition BC>
__global__ void updateSublatticeKernel(int* subLattice, int* opponent, bool isWhite, int size, int subSize, double temp, curandStatePhilox4_32_10_t *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= size * subSize) return;
    int nearSum = getNeighbourSum<BC>(opponent, id, size, subSize, isWhite);
    float deltaE = 2.*(float)subLattice[id] * (float)nearSum;
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

void reduceMagnetizationThrust(int* lattice, int lattLen, int* out){
    thrust::device_ptr<int> lattice_ptr(lattice);

    thrust::device_ptr<int> out_ptr(out); //out is passed already with +n_sweep
    
    thrust::constant_iterator<int> keys_first(0);
    thrust::constant_iterator<int> keys_last = keys_first + lattLen;

    thrust::reduce_by_key(keys_first, keys_last, lattice_ptr, thrust::make_discard_iterator(), out_ptr);

}

template<BoundaryCondition BC>
void blockSweepKernel(DataBlock* d, int blockSweeps, int n_blocks, int n_threads){
    for(int s = 0; s < blockSweeps; s++){
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->whites, d->blacks, true, d->size, d->subSize, d->temp, d->state);
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->blacks, d->whites, false, d->size, d->subSize, d->temp, d->state);
        //reduceMagnetizationKernel<<<n_blocks, n_threads>>>(
        //    d->whites, d->blacks, d->subLen, &d->magnetization[s]);
        reduceMagnetizationThrust(d->lattice, d->Len, &d->magnetization[s]);
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