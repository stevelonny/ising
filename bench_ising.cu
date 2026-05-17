#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "ising_gpu.hpp"
#include "ising_shared.hpp"

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

struct Options {
    string impl = "global";
    string bc = "Periodic";
    int size = 0;
    int sweeps = 0;
    int mag = 0;
    int repeats = 1;
};

static void print_usage(const char* prog) {
    cerr << "Usage: " << prog
         << " --impl {global,shared} --bc {Open,Periodic} --size N --sweeps S --mag {0,1} --repeats R\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need_value = [&](const string& name) {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << "\n";
                exit(1);
            }
            return string(argv[++i]);
        };

        if (arg == "--impl") opt.impl = need_value(arg);
        else if (arg == "--bc") opt.bc = need_value(arg);
        else if (arg == "--size") opt.size = stoi(need_value(arg));
        else if (arg == "--sweeps") opt.sweeps = stoi(need_value(arg));
        else if (arg == "--mag") opt.mag = stoi(need_value(arg));
        else if (arg == "--repeats") opt.repeats = stoi(need_value(arg));
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }

    if (opt.size <= 0 || opt.sweeps <= 0 || opt.repeats <= 0) {
        cerr << "--size, --sweeps, and --repeats must be positive\n";
        exit(1);
    }
    if (opt.impl != "global" && opt.impl != "shared") {
        cerr << "--impl must be global or shared\n";
        exit(1);
    }
    if (opt.bc != "Open" && opt.bc != "Periodic") {
        cerr << "--bc must be Open or Periodic\n";
        exit(1);
    }
    if (opt.mag != 0 && opt.mag != 1) {
        cerr << "--mag must be 0 or 1\n";
        exit(1);
    }
    return opt;
}

static inline BoundaryCondition parse_bc(const string& bc) {
    return (bc == "Open") ? Open : Periodic;
}

static inline int block_size_for_len(int len) {
    return min(256, max(64, len));
}

static vector<float> build_exp_table(double temp) {
    int deltaE[5] = {-8, -4, 0, 4, 8};
    vector<float> table(5);
    for (int i = 0; i < 5; ++i) {
        table[i] = expf(-(float)deltaE[i] / (float)temp);
    }
    table[2] = 1.0f;
    return table;
}

template <BoundaryCondition BC>
__global__ void updateTileKernelNoMag(int* lattice, int size, int color, int n_TilesRow, curandStatePhilox4_32_10_t* state) {
    int tile_id = blockIdx.x;
    int n_coloredTilesRow = n_TilesRow / 2;
    int tile_y = tile_id / n_coloredTilesRow;
    int tile_rowHalf = tile_id % n_coloredTilesRow;
    int tile_x = (tile_rowHalf * 2) + ((tile_y & 1) ^ color);
    int tileWidth = blockDim.x * 2;
    int tileHeight = blockDim.y;
    int globalXOrig = tile_x * tileWidth;
    int globalYOrig = tile_y * tileHeight;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;

    extern __shared__ int s_tile_raw[];
    int stride = tileWidth + 2;
    int total_elements = (tileWidth + 2) * (tileHeight + 2);

    for (int i = tid; i < total_elements; i += total_threads) {
        int sy = i / stride;
        int sx = i % stride;
        int gx = globalXOrig + sx - 1;
        int gy = globalYOrig + sy - 1;
        int val = 0;
        if (BC == Open) {
            if (gx >= 0 && gx < size && gy >= 0 && gy < size) {
                val = lattice[gy * size + gx];
            }
        } else {
            int wrapped_x = (gx + size) & (size - 1);
            int wrapped_y = (gy + size) & (size - 1);
            val = lattice[wrapped_y * size + wrapped_x];
        }
        s_tile_raw[sy * stride + sx] = val;
    }
    __syncthreads();

    int gy = globalYOrig + ty;
    int stateIdx = gy * (size / 2) + (globalXOrig + 2 * tx + (gy & 1)) / 2;
    curandStatePhilox4_32_10_t threadState = state[stateIdx];

    int sy = ty + 1;
    int sx_w = 2 * tx + (gy & 1) + 1;
    int sum_w = s_tile_raw[(sy - 1) * stride + sx_w] + s_tile_raw[(sy + 1) * stride + sx_w] + s_tile_raw[sy * stride + sx_w - 1] + s_tile_raw[sy * stride + sx_w + 1];
    int spin_w = s_tile_raw[sy * stride + sx_w];
    int exp_idx_w = ((2 * spin_w * sum_w) + 8) / 4;
    if (curand_uniform(&threadState) < d_expTable[exp_idx_w]) {
        spin_w *= -1;
        s_tile_raw[sy * stride + sx_w] = spin_w;
    }
    __syncthreads();

    int sx_b = 2 * tx + (1 - (gy & 1)) + 1;
    int sum_b = s_tile_raw[(sy - 1) * stride + sx_b] + s_tile_raw[(sy + 1) * stride + sx_b] + s_tile_raw[sy * stride + sx_b - 1] + s_tile_raw[sy * stride + sx_b + 1];
    int spin_b = s_tile_raw[sy * stride + sx_b];
    int exp_idx_b = ((2 * spin_b * sum_b) + 8) / 4;
    if (curand_uniform(&threadState) < d_expTable[exp_idx_b]) {
        spin_b *= -1;
        s_tile_raw[sy * stride + sx_b] = spin_b;
    }
    __syncthreads();

    state[stateIdx] = threadState;
    __syncthreads();

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

template <BoundaryCondition BC>
void run_global_no_mag(int* d_lattice, int size, int sweeps, curandStatePhilox4_32_10_t* d_state) {
    int subSize = size / 2;
    int subLen = size * subSize;
    int nthreads = block_size_for_len(subLen);
    int nblocks = (subLen + nthreads - 1) / nthreads;

    DataBlock d{};
    d.lattice = d_lattice;
    d.whites = d_lattice;
    d.blacks = d_lattice + subLen;
    d.size = size;
    d.Len = size * size;
    d.subLen = subLen;
    d.subSize = subSize;
    d.state = d_state;
    d.magnetization = nullptr;

    for (int s = 0; s < sweeps; ++s) {
        updateSublatticeKernel<BC><<<nblocks, nthreads>>>(d.whites, d.blacks, true, d.size, d.subSize, d.state);
        updateSublatticeKernel<BC><<<nblocks, nthreads>>>(d.blacks, d.whites, false, d.size, d.subSize, d.state);
    }
}

template <BoundaryCondition BC>
void run_shared_no_mag(int* d_lattice, int size, int sweeps, curandStatePhilox4_32_10_t* d_state) {
    const int TILE_DIM_X = 16;
    const int TILE_DIM_Y = 16;
    dim3 block(TILE_DIM_X / 2, TILE_DIM_Y);
    int n_TilesRow = size / TILE_DIM_X;
    int total_tiles = n_TilesRow * (size / TILE_DIM_Y);
    dim3 grid(total_tiles / 2);
    int tileWidth = TILE_DIM_X + 2;
    int tileHeight = TILE_DIM_Y + 2;
    size_t shared_mem_bytes = (tileWidth * tileHeight) * sizeof(int);

    for (int s = 0; s < sweeps; ++s) {
        updateTileKernelNoMag<BC><<<grid, block, shared_mem_bytes>>>(d_lattice, size, 0, n_TilesRow, d_state);
        updateTileKernelNoMag<BC><<<grid, block, shared_mem_bytes>>>(d_lattice, size, 1, n_TilesRow, d_state);
    }
}

template <BoundaryCondition BC>
void run_shared_with_mag(int* d_lattice, int size, int sweeps, int* d_mag, curandStatePhilox4_32_10_t* d_state) {
    const int TILE_DIM_X = 16;
    const int TILE_DIM_Y = 16;
    dim3 block(TILE_DIM_X / 2, TILE_DIM_Y);
    int n_TilesRow = size / TILE_DIM_X;
    int total_tiles = n_TilesRow * (size / TILE_DIM_Y);
    dim3 grid(total_tiles / 2);
    int tileWidth = TILE_DIM_X + 2;
    int tileHeight = TILE_DIM_Y + 2;
    size_t shared_mem_bytes = (tileWidth * tileHeight) * sizeof(int);

    for (int s = 0; s < sweeps; ++s) {
        updateTileKernel<BC><<<grid, block, shared_mem_bytes>>>(d_lattice, size, 0, n_TilesRow, &d_mag[s], d_state);
        updateTileKernel<BC><<<grid, block, shared_mem_bytes>>>(d_lattice, size, 1, n_TilesRow, &d_mag[s], d_state);
    }
}

static void init_lattice_and_state(int* d_lattice, int lattice_len, curandStatePhilox4_32_10_t* d_state, int state_len) {
    int init_threads = 256;
    int init_blocks = (max(lattice_len, state_len) + init_threads - 1) / init_threads;
    initCurand<<<init_blocks, init_threads>>>(d_state, 42UL);
    initSublatticeKernel<<<(lattice_len + init_threads - 1) / init_threads, init_threads>>>(d_lattice, lattice_len, d_state);
}

static void print_json_line(const string& impl, const string& bc, int size, int sweeps, int mag, int repeat_index, float elapsed_ms, int final_mag, bool include_mag) {
    cout << "{"
         << "\"impl\":\"" << impl << "\""
         << ",\"bc\":\"" << bc << "\""
         << ",\"size\":" << size
         << ",\"sweeps\":" << sweeps
         << ",\"mag\":" << mag
         << ",\"repeat_index\":" << repeat_index
         << ",\"elapsed_ms\":" << elapsed_ms;
    if (include_mag) {
        cout << ",\"final_magnetization\":" << final_mag;
    }
    cout << "}" << endl;
}

template <typename Runner>
static void run_repeats(const Options& opt, Runner runner) {
    BoundaryCondition bc = parse_bc(opt.bc);
    vector<float> expTable = build_exp_table(2.269);
    CUDA_CHECK(cudaMemcpyToSymbol(d_expTable, expTable.data(), expTable.size() * sizeof(float)));

    int lattice_len = opt.size * opt.size;
    int state_len = lattice_len;

    for (int rep = 0; rep < opt.repeats; ++rep) {
        int* d_lattice = nullptr;
        curandStatePhilox4_32_10_t* d_state = nullptr;
        CUDA_CHECK(cudaMalloc(&d_lattice, lattice_len * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_state, state_len * sizeof(curandStatePhilox4_32_10_t)));
        init_lattice_and_state(d_lattice, lattice_len, d_state, state_len);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Warmup: one short untimed pass.
        runner(d_lattice, d_state, bc, max(1, opt.sweeps / 10), false, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        int* d_mag = nullptr;
        if (opt.mag) {
            CUDA_CHECK(cudaMalloc(&d_mag, opt.sweeps * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_mag, 0, opt.sweeps * sizeof(int)));
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        runner(d_lattice, d_state, bc, opt.sweeps, opt.mag != 0, d_mag);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        int final_magnetization = 0;
        if (opt.mag) {
            CUDA_CHECK(cudaMemcpy(&final_magnetization, d_mag + (opt.sweeps - 1), sizeof(int), cudaMemcpyDeviceToHost));
        }

        print_json_line(opt.impl, opt.bc, opt.size, opt.sweeps, opt.mag, rep, elapsed_ms, final_magnetization, opt.mag != 0);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        if (d_mag) CUDA_CHECK(cudaFree(d_mag));
        CUDA_CHECK(cudaFree(d_lattice));
        CUDA_CHECK(cudaFree(d_state));
    }
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    auto runner = [&](int* d_lattice, curandStatePhilox4_32_10_t* d_state, BoundaryCondition bc, int sweeps, bool with_mag, int* d_mag) {
        int size = opt.size;
        if (opt.impl == "global") {
            int subSize = size / 2;
            int subLen = size * subSize;
            int nthreads = block_size_for_len(subLen);
            int nblocks = (subLen + nthreads - 1) / nthreads;
            if (with_mag) {
                DataBlock d{};
                d.lattice = d_lattice;
                d.whites = d_lattice;
                d.blacks = d_lattice + subLen;
                d.size = size;
                d.Len = size * size;
                d.subLen = subLen;
                d.subSize = subSize;
                d.state = d_state;
                d.magnetization = d_mag;
                if (bc == Open) {
                    blockSweepKernel<Open>(&d, sweeps, nblocks, nthreads);
                } else {
                    blockSweepKernel<Periodic>(&d, sweeps, nblocks, nthreads);
                }
            } else {
                if (bc == Open) {
                    run_global_no_mag<Open>(d_lattice, size, sweeps, d_state);
                } else {
                    run_global_no_mag<Periodic>(d_lattice, size, sweeps, d_state);
                }
            }
        } else {
            if (with_mag) {
                    if (bc == Open) {
                        run_shared_with_mag<Open>(d_lattice, size, sweeps, d_mag, d_state);
                    } else {
                        run_shared_with_mag<Periodic>(d_lattice, size, sweeps, d_mag, d_state);
                    }
            } else {
                if (bc == Open) {
                    run_shared_no_mag<Open>(d_lattice, size, sweeps, d_state);
                } else {
                    run_shared_no_mag<Periodic>(d_lattice, size, sweeps, d_state);
                }
            }
        }
    };

    run_repeats(opt, runner);
    return 0;
}
