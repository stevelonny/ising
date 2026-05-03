#include "ising_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
//#include <algorithm>

using namespace std;

#define MAX_SWEEPS 10000

double compute_mean(int* data, int n){
    double mean{0.};
    for(int i{0}; i<n; i++){
        mean += data[i];
    }
    return mean / (double)n;
}

double compute_mean(double* data, int n){
    double mean{0.};
    for(int i{0}; i<n; i++){
        mean += data[i];
    }
    return mean / (double)n;
}

double compute_var(int* data, int n, double mean){
    double var{0.};
    for(int i{0}; i<n; i++){
        var += (data[i] - mean) * (data[i] - mean);
    }
    return var / (double)n;
}

int main(int argc, char** argv){

    if(argc != 5){ 
        cerr << "Usage: " << argv[0] << " <grid_size> <temperature> <nblocksweeps> <blocksize>" << endl;
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
    int nblocksweeps = stoi(argv[3]);
    if (nblocksweeps <= 0) {
        cerr << "Number of blocksweeps must be positive." << endl;
        return -1;
    }
    int blocksize = stoi(argv[4]);
    if (blocksize <= 0) {
        cerr << "Block size must be positive." << endl;
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

    /* annealing */
    //useful when using open border condition
    int sweep{0};
    int index{0};
    int window_size{500};
    int* d_magnetization;
    cudaMalloc(&d_magnetization, window_size * sizeof(int));
    cudaMemset(d_magnetization, 0, window_size * sizeof(int));
    int* magnetization = new int[window_size];

    //temperature annealing schedule: cool from high T to target over 1000 sweeps
    int anneal_sweeps = 5000;
    double T_start{5.0};
    double T_current{5.0};
    std::cout << "Annealing phase (annealing T: " << T_start << " -> " << temp << ")" << endl;
    while(sweep < anneal_sweeps){
        T_current = T_start - (T_start - temp) * ((double)sweep / (double)(anneal_sweeps - 1));
        
        //perform mc sweep
        updateSublatticeKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, T_current, d_state);
        updateSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, T_current, d_state);
        //compute magnetization
        index = sweep%window_size;
        cudaMemset(&d_magnetization[index], 0, sizeof(int));
        reduceMagnetizationKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, subLen, &d_magnetization[index]);

        if(sweep%100 == 0){
            cudaMemcpy(&magnetization[index], &d_magnetization[index], sizeof(int), cudaMemcpyDeviceToHost);
            cout  << "Sweep " << sweep << "\tT: " << T_current << "\tMag: " << magnetization[index] << endl;
        }

        sweep++;
    }
    cudaMemcpy(&magnetization[index], &d_magnetization[index], sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Sweep " << sweep << "\tT: " << T_current << "\tMag: " << magnetization[index] << endl;
    cout << "Resetting sweep counter." << endl;
    
    /* thermalization */
    //check convergence only after cooling completes
    sweep = 0;
    bool is_thermalized{false};
    cout << "Thermalization phase" << endl;
    while(!is_thermalized && sweep < MAX_SWEEPS){
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
            //diff and threshold
            double diff = fabs(mean1 - mean2);
            double threshold = 0.05 * sqrt(var1 + var2);
            if(diff < threshold && (var1 > 1.0 || var2 > 1.0)){//zero variance can mean frozen system
                is_thermalized = true;
                cout << "Thermalization achieved at sweep " << sweep << endl;
                break;
            }
        }
        sweep++;        
    }
    cout << "Final thermalization: " << (is_thermalized ? "SUCCESS" : "INCOMPLETE") << endl;

    /* interlude */
    cudaFree(d_magnetization);
    DataBlock d;
    d.whites = d_whites;
    d.blacks = d_blacks;
    d.size = size;
    d.subSize = subSize;
    d.subLen = subLen;
    d.temp = temp;
    d.state = d_state;
    cudaMalloc(&d.magnetization, blocksize * sizeof(int));
    cudaMemset(d.magnetization, 0, blocksize * sizeof(int));

    /* simulation/measure */
    //may need just one block but oh well
    for(int b{0}; b<nblocksweeps; b++){
        //single block sweep
        for(int s = 0; s < blocksize; s++){
            updateSublatticeKernel<<<nblocks, nthreads>>>(
                d.whites, d.blacks, true, d.size, d.subSize, d.temp, d.state);
            updateSublatticeKernel<<<nblocks, nthreads>>>(
                d.blacks, d.whites, false, d.size, d.subSize, d.temp, d.state);
            reduceMagnetizationKernel<<<nblocks, nthreads>>>(
                d.whites, d.blacks, d.subLen, &d.magnetization[s]);
        }
        //copy magnetization values
        cudaMemcpy(magnetization, d.magnetization, blocksize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(d.magnetization, 0, blocksize * sizeof(int));
        double mean = compute_mean(magnetization, blocksize);
        double var = compute_var(magnetization, blocksize, mean);
        cout << "Block #" << b << "\t" << "Mag: " << mean << "\t" << "std: " << sqrt(var) << endl;
    }

    //cleanup
    cudaFree(d_whites);
    cudaFree(d_blacks);
    cudaFree(d_magnetization);
    cudaFree(d_state);
    delete[] magnetization;

    return 0;
}