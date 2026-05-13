#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

using namespace std;

__constant__ float d_expTable[5];

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
};

enum UpdateImplementation{
    GlobalMemory,
    SharedMemory
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
__global__ void updateSublatticeKernel(int* subLattice, int* opponent, bool isWhite, int size, int subSize, curandStatePhilox4_32_10_t *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= size * subSize) return;
    int nearSum = getNeighbourSum<BC>(opponent, id, size, subSize, isWhite);
    int deltaE = 2 * subLattice[id] * nearSum;
    int exp_idx = (deltaE + 8) / 4;
    float r = curand_uniform(&state[id]);
    if(r < d_expTable[exp_idx]){
        subLattice[id] *= -1;
    }
}

template<BoundaryCondition BC>
__device__ inline int readWithBoundary(const int* lattice, int row, int col, int size, int subSize, bool sizePow2, bool subSizePow2){
    if(BC == Open){
        if(row < 0 || row >= size || col < 0 || col >= subSize){
            return 0;
        }
        return lattice[row * subSize + col];
    }
    int wrappedRow = sizePow2 ? (row & (size - 1)) : ((row % size + size) % size);
    int wrappedCol = subSizePow2 ? (col & (subSize - 1)) : ((col % subSize + subSize) % subSize);
    return lattice[wrappedRow * subSize + wrappedCol];
}

template<BoundaryCondition BC>
__global__ void updateSublatticeSharedKernel(int* subLattice, const int* opponent, bool isWhite, int size, int subSize, curandStatePhilox4_32_10_t *state, int* magnetization){
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
    int tileWidth = blockDim.x + 2;
    int localIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int tileIndex = ly * tileWidth + lx;
    bool valid = (gy < size && gx < subSize);
    bool sizePow2 = (size & (size - 1)) == 0;
    bool subSizePow2 = (subSize & (subSize - 1)) == 0;

    extern __shared__ int shared[];
    int* shSub = shared;
    int* shOpp = shSub + (blockDim.x + 2) * (blockDim.y + 2);
    __shared__ int blockSum;
    if(localIndex == 0){
        blockSum = 0;
    }
    __syncthreads();

    shSub[tileIndex] = readWithBoundary<BC>(subLattice, gy, gx, size, subSize, sizePow2, subSizePow2);
    shOpp[tileIndex] = readWithBoundary<BC>(opponent, gy, gx, size, subSize, sizePow2, subSizePow2);

    if(threadIdx.x == 0){
        shSub[ly * tileWidth] = readWithBoundary<BC>(subLattice, gy, gx - 1, size, subSize, sizePow2, subSizePow2);
        shOpp[ly * tileWidth] = readWithBoundary<BC>(opponent, gy, gx - 1, size, subSize, sizePow2, subSizePow2);
    }
    if(threadIdx.x == blockDim.x - 1){
        shSub[ly * tileWidth + (blockDim.x + 1)] = readWithBoundary<BC>(subLattice, gy, gx + 1, size, subSize, sizePow2, subSizePow2);
        shOpp[ly * tileWidth + (blockDim.x + 1)] = readWithBoundary<BC>(opponent, gy, gx + 1, size, subSize, sizePow2, subSizePow2);
    }
    if(threadIdx.y == 0){
        shSub[lx] = readWithBoundary<BC>(subLattice, gy - 1, gx, size, subSize, sizePow2, subSizePow2);
        shOpp[lx] = readWithBoundary<BC>(opponent, gy - 1, gx, size, subSize, sizePow2, subSizePow2);
    }
    if(threadIdx.y == blockDim.y - 1){
        shSub[(blockDim.y + 1) * tileWidth + lx] = readWithBoundary<BC>(subLattice, gy + 1, gx, size, subSize, sizePow2, subSizePow2);
        shOpp[(blockDim.y + 1) * tileWidth + lx] = readWithBoundary<BC>(opponent, gy + 1, gx, size, subSize, sizePow2, subSizePow2);
    }
    if(threadIdx.x == 0 && threadIdx.y == 0){
        shSub[0] = readWithBoundary<BC>(subLattice, gy - 1, gx - 1, size, subSize, sizePow2, subSizePow2);
        shOpp[0] = readWithBoundary<BC>(opponent, gy - 1, gx - 1, size, subSize, sizePow2, subSizePow2);
    }
    if(threadIdx.x == blockDim.x - 1 && threadIdx.y == 0){
        shSub[blockDim.x + 1] = readWithBoundary<BC>(subLattice, gy - 1, gx + 1, size, subSize, sizePow2, subSizePow2);
        shOpp[blockDim.x + 1] = readWithBoundary<BC>(opponent, gy - 1, gx + 1, size, subSize, sizePow2, subSizePow2);
    }
    if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1){
        shSub[(blockDim.y + 1) * tileWidth] = readWithBoundary<BC>(subLattice, gy + 1, gx - 1, size, subSize, sizePow2, subSizePow2);
        shOpp[(blockDim.y + 1) * tileWidth] = readWithBoundary<BC>(opponent, gy + 1, gx - 1, size, subSize, sizePow2, subSizePow2);
    }
    if(threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1){
        shSub[(blockDim.y + 1) * tileWidth + (blockDim.x + 1)] = readWithBoundary<BC>(subLattice, gy + 1, gx + 1, size, subSize, sizePow2, subSizePow2);
        shOpp[(blockDim.y + 1) * tileWidth + (blockDim.x + 1)] = readWithBoundary<BC>(opponent, gy + 1, gx + 1, size, subSize, sizePow2, subSizePow2);
    }
    __syncthreads();

    int updatedSpin = 0;
    if(valid){
        bool rowParity = (gy & 1) != 0;
        if(isWhite){
            rowParity = !rowParity;
        }
        int nearSum = shOpp[(ly - 1) * tileWidth + lx] + shOpp[(ly + 1) * tileWidth + lx] + shOpp[tileIndex]
                    + (rowParity ? shOpp[ly * tileWidth + (lx - 1)] : shOpp[ly * tileWidth + (lx + 1)]);
        int deltaE = 2 * shSub[tileIndex] * nearSum;
        int exp_idx = (deltaE + 8) / 4;
        int id = gy * subSize + gx;
        float r = curand_uniform(&state[id]);
        updatedSpin = (r < d_expTable[exp_idx]) ? -shSub[tileIndex] : shSub[tileIndex];
        subLattice[id] = updatedSpin;
    }

    if(valid){
        atomicAdd(&blockSum, updatedSpin);
    }
    __syncthreads();
    if(localIndex == 0){
        atomicAdd(magnetization, blockSum);
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
void blockSweepKernel(DataBlock* d, int blockSweeps, int n_blocks, int n_threads, UpdateImplementation implementation = GlobalMemory){
    if(implementation == SharedMemory){
        dim3 threads(32, 16);
        dim3 blocks((d->subSize + threads.x - 1) / threads.x, (d->size + threads.y - 1) / threads.y);
        size_t shmemBytes = 2 * (threads.x + 2) * (threads.y + 2) * sizeof(int);
        cudaMemset(d->magnetization, 0, blockSweeps * sizeof(int));
        for(int s = 0; s < blockSweeps; s++){
            updateSublatticeSharedKernel<BC><<<blocks, threads, shmemBytes>>>(
                d->whites, d->blacks, true, d->size, d->subSize, d->state, &d->magnetization[s]);
            updateSublatticeSharedKernel<BC><<<blocks, threads, shmemBytes>>>(
                d->blacks, d->whites, false, d->size, d->subSize, d->state, &d->magnetization[s]);
        }
        return;
    }
    for(int s = 0; s < blockSweeps; s++){
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->whites, d->blacks, true, d->size, d->subSize, d->state);
        updateSublatticeKernel<BC><<<n_blocks, n_threads>>>(
            d->blacks, d->whites, false, d->size, d->subSize, d->state);
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
