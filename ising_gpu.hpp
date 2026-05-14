#include <random>
#include <cmath>
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

//this will update both black and white in shared memory and perform redution for magnetization
template<BoundaryCondition BC>
__global__ void updateSharedSublatticeKernel(int* subLattice, int* opponent, bool isWhite, int size, int subSize, int* magnetization, curandStatePhilox4_32_10_t *state){
    extern __shared__ int shared[]; //unique dyn mem pointer
    //both sublattices will need a halo of 1 around the block for neighbour access
    int* shOpp = shared; //only the opponent's sublattice is needed for neighbour access
    int* shReduce = shared + (blockDim.x + 2) * (blockDim.y + 2); //after opp's subalattice

    //global indeces for accessing the original lattice
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = gy * subSize + gx; //1d global index
    //local indeces for accessing shared memory
    int lx = threadIdx.x + 1; //+1 for halo
    int ly = threadIdx.y + 1;
    int local_idx = ly * (blockDim.x + 2) + lx;

    int tile_width = blockDim.x + 2; //including halo
    int tile_height = blockDim.y + 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; //1d thread id

    int num_threads = blockDim.x * blockDim.y;
    int tile_size = blockDim.x * blockDim.y;

    /* load data into shared memory with halo */
    //remember corners
    //remember boundary conditions
    for(int i=tid; i<tile_size; i+=num_threads){
        int r_sh = i / tile_width; //row in shared memory
        int c_sh = i % tile_width; //col in shared memory
        //global coords with halo offset
        int r_gl = blockIdx.y * blockDim.y + r_sh - 1;
        int c_gl = blockIdx.x * blockDim.x + c_sh - 1;
        int val = 0;
        //handle obc
        if(BC == Open){
            if (r_gl >= 0 && r_gl < size && c_gl >= 0 && c_gl < subSize){
            val = opponent[r_gl * subSize + c_gl];
            }
        }
        //pbc: wrap indices using bitwise AND
        else if(BC == Periodic){
            r_gl = (r_gl + size) & (size - 1);
            c_gl = (c_gl + subSize) & (subSize - 1);
            val = opponent[r_gl * subSize + c_gl];
        }
        shOpp[r_sh * tile_width + c_sh] = val;
    }
    __syncthreads();

    /* update */
    if(gx < subSize && gy < size){
        //up + down + self(left/right)
        int nearSum = shOpp[(ly - 1) * tile_width + lx] + shOpp[(ly + 1) * tile_width + lx] + shOpp[ly * tile_width + lx];
        //left and right switch places for each row's parity
        bool rowParity = (gy % 2 == 0);
        bool parityAdjusted = isWhite ? rowParity : !rowParity;
        nearSum += parityAdjusted ? shOpp[ly * tile_width + lx - 1] : shOpp[ly * tile_width + lx + 1];

        int spin = subLattice[global_idx];
        int deltaE = 2 * spin * nearSum;
        int exp_idx = (deltaE + 8) / 4;
        float r = curand_uniform(&state[global_idx]);
        if(r < d_expTable[exp_idx]){
            subLattice[global_idx] *= -1;
            spin *= -1;
        }
        shReduce[tid] = spin;
    }
    else{
        shReduce[tid] = 0;
    }
    __syncthreads();
    /* reduction in shared memory */
    for(int s = tile_size / 2; s > 0; s /= 2){
        if(tid < s){
            shReduce[tid] += shReduce[tid + s];
        }
    }
    /* write back result */
    if(tid == 0){
        atomicAdd(magnetization, shReduce[0]);
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