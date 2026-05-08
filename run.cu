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

    if(argc != 5 && argc != 6){ 
        cerr << "Usage: " << argv[0] << " <power of 2 for grid_size> <temperature> <nblocksweeps> <blocksize> [obc|pbc]" << endl;
        return -1;
    }
    int power2 = stoi(argv[1]);
    if(power2 <= 0){
        cerr << "Power of 2 for grid size must be a positive integer." << endl;
        return -1;
    }
    double temp = stod(argv[2]);
    if(temp <= 0.0){
        cerr << "Temperature must be positive." << endl;
        return -1;
    }
    int nblocksweeps = stoi(argv[3]);
    if(nblocksweeps <= 0){
        cerr << "Number of blocksweeps must be positive." << endl;
        return -1;
    }
    int blocksize = stoi(argv[4]);
    if(blocksize <= 0){
        cerr << "Block size must be positive." << endl;
        return -1;
    }
    bool obc = false;
    if(argc == 6){
        if(string(argv[5]) == "obc"){
            obc = true;
        }
        else if(string(argv[5]) == "pbc"){
            obc = false;
        }
        else{
            cerr << "Invalid boundary condition. Use 'obc' or 'pbc'." << endl;
            return -1;
        }
    }

    /* setup */
    int size = 1 << power2; // grid size is 2^power2
    cout << "Grid size: " << size << "x" << size << endl;
    int subSize = size/2; //numbers of columns for each sublattice
    int subLen = size*subSize; //length of each sublattice array (total number of elements)
    int Len = 2*subLen; //total number of elements in the lattice
    //determine grid and block dimensions
    int nthreads = subLen;
    if(subLen > 1024) nthreads = 1024;
    int nblocks = (subLen + nthreads - 1) / nthreads;
    cout << "Launching kernel with " << nblocks << " blocks and " << nthreads << " threads per block." << endl;
    //allocate arrays on gpu
    //allocate a contiguous block for both sublattices (useful for thrust reduction)
    int* d_lattice;
    cudaMalloc(&d_lattice, 2 * subLen * sizeof(int));
    int* d_whites = d_lattice;
    int* d_blacks = d_lattice + subLen;
    //init curand state
    std::random_device rd;
    //int rd{42};
    curandStatePhilox4_32_10_t* d_state;
    cudaMalloc(&d_state, nblocks * nthreads * sizeof(curandStatePhilox4_32_10_t));
    initCurand<<<nblocks, nthreads>>>(d_state, rd());
    //initialize sublattices on gpu
    initSublatticeKernel<<<nblocks, nthreads>>>(d_whites, subLen, d_state);
    initSublatticeKernel<<<nblocks, nthreads>>>(d_blacks, subLen, d_state);
    cudaDeviceSynchronize();

    int sweep{0};
    int index{0};
    int window_size{500};
    int* d_magnetization;
    cudaMalloc(&d_magnetization, window_size * sizeof(int));
    cudaMemset(d_magnetization, 0, window_size * sizeof(int));
    int* magnetization = new int[window_size];
    
    /* annealing */
    //useful when using open border condition
    if(obc==true){ //temperature annealing schedule: cool from high T to target over 1000 sweeps
        int anneal_sweeps = 1000;
        double T_start{5.0};
        double T_current{5.0};
        std::cout << "Annealing phase (annealing T: " << T_start << " -> " << temp << ")" << endl;
        while(sweep < anneal_sweeps){
            T_current = T_start - (T_start - temp) * ((double)sweep / (double)(anneal_sweeps - 1));
            
            //perform mc sweep
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, T_current, d_state);
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, T_current, d_state);
            //compute magnetization
            index = sweep%window_size;
            cudaMemset(&d_magnetization[index], 0, sizeof(int));
            reduceMagnetizationKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, subLen, &d_magnetization[index]);

            if(sweep%100 == 0){
                cudaMemcpy(&magnetization[index], &d_magnetization[index], sizeof(int), cudaMemcpyDeviceToHost);
                std::cout  << "Sweep " << sweep << "\tT: " << T_current << "\tMag: " << magnetization[index] << endl;
            }

            sweep++;
        }
        cudaMemcpy(&magnetization[index], &d_magnetization[index], sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Sweep " << sweep << "\tT: " << T_current << "\tMag: " << magnetization[index] << endl;
        std::cout << "Resetting sweep counter." << endl;
        sweep = 0;
    }
    /* thermalization */
    //check convergence only after cooling completes
    double mean1{0.};
    double mean2{0.};
    double var1{0.};
    double var2{0.};
    bool is_thermalized{false};
    std::cout << "Thermalization phase" << endl;
    while(!is_thermalized && sweep < MAX_SWEEPS){
        //perform mc sweep
        if(obc){
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, temp, d_state);
            updateSublatticeKernel<Open><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, temp, d_state);
        }
        else{
            updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_whites, d_blacks, true, size, subSize, temp, d_state);
            updateSublatticeKernel<Periodic><<<nblocks, nthreads>>>(d_blacks, d_whites, false, size, subSize, temp, d_state);
        }
        //compute magnetization
        index = sweep%window_size;
        cudaMemset(&d_magnetization[index], 0, sizeof(int));
        reduceMagnetizationKernel<<<nblocks, nthreads>>>(d_whites, d_blacks, subLen, &d_magnetization[index]);

        if(sweep>=window_size && sweep%100 == 0){
            //retrieve d_magnetization
            cudaMemcpy(magnetization, d_magnetization, window_size * sizeof(int), cudaMemcpyDeviceToHost);
            //calculate mean and var for 2 consecutive subwindows
            mean1 = 0.;
            mean2 = 0.;
            var1 = 0.;
            var2 = 0.;
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
            if(diff < threshold /* && (var1 > 1.0 || var2 > 1.0) */){//zero variance can mean frozen system
                is_thermalized = true;
                cout << "Thermalization achieved at sweep " << sweep << endl;
                break;
            }
        }
        sweep++;        
    }
    cout << "Final thermalization: " << (is_thermalized ? "SUCCESS" : "INCOMPLETE") << endl;
    //cout << "values: " << mean1 << "+-" << sqrt(var1) << "\t" << mean2 << "+-" << sqrt(var2) << " diff: " << fabs(mean1-mean2) << " threshold: " << 0.05 * sqrt(var1 + var2) << endl;

    /* interlude */
    cudaFree(d_magnetization);
    DataBlock d;
    d.whites = d_whites;
    d.blacks = d_blacks;
    d.size = size;
    d.Len = Len;
    d.subSize = subSize;
    d.subLen = subLen;
    d.temp = temp;
    d.state = d_state;
    cudaMalloc(&d.magnetization, blocksize * sizeof(int));
    cudaMemset(d.magnetization, 0, blocksize * sizeof(int));
    //cudaMalloc(&d.magnetization_thrust, blocksize * sizeof(int));
    //cudaMemset(d.magnetization_thrust, 0, blocksize * sizeof(int));

    /* simulation/measure */
    //may need just one block but oh well
    for(int b{0}; b<nblocksweeps; b++){
        //single block sweep
        if(obc){
            blockSweepKernel<Open>(&d, blocksize, nblocks, nthreads);
        }
        else{
            blockSweepKernel<Periodic>(&d, blocksize, nblocks, nthreads);
        }
        //copy magnetization values
        cudaMemcpy(magnetization, d.magnetization, blocksize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(d.magnetization, 0, blocksize * sizeof(int));
        double mean = compute_mean(magnetization, blocksize);
        double var = compute_var(magnetization, blocksize, mean);
        //copy thrust magnetization values
        //cudaMemcpy(magnetization, d.magnetization_thrust, sizeof(int), cudaMemcpyDeviceToHost);
        //double thrust_mean = compute_mean(magnetization, blocksize);
        //double thrust_var = compute_var(magnetization, blocksize, thrust_mean);
        cout << "Block #" << b << "\t" << "Mag: " << mean << "\t" << "std: " << sqrt(var) << endl;
        //cout << "Block #" << b << "\t" << "Mag: " << thrust_mean << "\t" << "std: " << sqrt(thrust_var) << endl;
    }

    /* cleanup */
    //cudaFree(d_whites);
    //cudaFree(d_blacks);
    cudaFree(d_lattice);
    cudaFree(d_magnetization);
    cudaFree(d_state);
    delete[] magnetization;

    return 0;
}