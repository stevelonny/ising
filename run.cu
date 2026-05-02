#include "ising_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

using namespace std;

#define MAX_SWEEPS 10000

int main(int argc, char** argv){

    if(argc != 4){ 
        cerr << "Usage: " << argv[0] << " <grid_size> <temperature> <sweeps>" << endl;
        return -1;
    }
    int size = stoi(argv[1]);
    if (size % 2 != 0 || size <= 0) {
        cerr << "Grid size must be a positive even integer." << endl;
        return -1;
    }
    double temp = stod(argv[2]);
    if (temp <= 0.0) {
        cerr << "Temperature must be positive." << endl;
        return -1;
    }
    int sweeps = stoi(argv[3]);
    if (sweeps <= 0) {
        cerr << "Number of sweeps must be positive." << endl;
        return -1;
    }

    /* setup */
    int subSize = size/2; //numbers of columns for each sublattice
    int subLen = size*subSize; //length of each sublattice array (total number of elements)
    //determine grid and block dimensions
    int nthreads = subLen;
    if(subLen > 1024) nthreads = 1024;
    int nblocks = (subLen + nthreads - 1) / nthreads;
    cout << "Launching kernel with " << nblocks << " blocks and " << nthreads << " threads per block." << endl;
    //allocate arrays on gpu
    int* d_whites;
    int* d_blacks;
    cudaMalloc(&d_whites, subLen * sizeof(int));
    cudaMalloc(&d_blacks, subLen * sizeof(int));
    //init curand state
    std::random_device rd;
    //int rd{42};
    curandState* d_state;
    cudaMalloc(&d_state, nblocks * nthreads * sizeof(curandState));
    initCurand<<<nblocks, nthreads>>>(d_state, rd());
    //initialize sublattices on gpu
    initSublatticeKernel<<<nblocks, nthreads>>>(d_whites, subLen, d_state);
    initSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, subLen, d_state);
    cudaDeviceSynchronize();

    /* thermalization */
    int window_size{500};
    bool is_thermalized{false};
    int sweep{0};
    int index{0};
    int* d_magnetization;
    cudaMalloc(&d_magnetization, window_size * sizeof(int));
    cudaMemset(d_magnetization, 0, window_size * sizeof(int));
    int* magnetization = new int[window_size];

    while(!is_thermalized && sweep<MAX_SWEEPS){
        //perform mc sweep
        updateSublatticeKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, temp, d_state);
        updateSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, temp, d_state);
        //compute magnetization
        index = sweep%window_size;
        cudaMemset(&d_magnetization[index], 0, sizeof(int));
        reduceMagnetizationKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, subLen, &d_magnetization[index]);

        if(sweep>=window_size && sweep%100 == 0){
            //retrieve d_magnetization
            cudaMemcpy(magnetization, d_magnetization, window_size * sizeof(int), cudaMemcpyDeviceToHost);
            //calculate mean and var for 2 consecutive subwindows
            double mean1{0.};
            double mean2{0.};
            double var1{0.};
            double var2{0.};
            for(int i{0}; i<window_size/2; i++){
                mean1 += magnetization[i];
                mean2 += magnetization[i + window_size/2];
            }
            mean1 /= (double)window_size/2.;
            mean2 /= (double)window_size/2.;
            for(int i{0}; i<window_size/2; i++){
                var1 += (magnetization[i] - mean1) * (magnetization[i] - mean1);
                var2 += (magnetization[i + window_size/2] - mean2) * (magnetization[i + window_size/2] - mean2);
            }
            var1 /= (double)window_size/2.;
            var2 /= (double)window_size/2.;
            //diff and thereshold
            double diff = abs(mean1 - mean2);
            double threshold = 0.05 * sqrt(var1 + var2);
            if(diff < threshold){
                is_thermalized = true;
                cout << "Thermalization achieved at sweep " << sweep << "." << endl;
            }
            //reset d_magnetization
            cudaMemset(d_magnetization, 0, window_size * sizeof(int));
        }
        sweep++;        
    }
    if(!is_thermalized){
        cout << "Thermalization not achieved after " << MAX_SWEEPS << " sweeps." << endl;
    }

    /* simulation */
    

    //cleanup
    cudaFree(d_whites);
    cudaFree(d_blacks);
    cudaFree(d_magnetization);
    cudaFree(d_state);
    delete[] magnetization;

    return 0;
}