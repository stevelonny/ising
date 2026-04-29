/* benchmark CPU, without output */
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include "ising_cpu.hpp"

using namespace std;

int main(int argc, char** argv){

    if(argc != 4){ 
        cerr << "Usage: " << argv[0] << " <grid_size> <temperature> <nsweeps>" << endl;
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
    int nsweeps = stoi(argv[3]);
    if (nsweeps <= 0) {
        cerr << "Number of sweeps must be positive." << endl;
        return -1;
    }

    int subSize = size/2; //numbers of columns for each sublattice
    int subLen = size*subSize; //length of each sublattice array (total number of elements)

    auto start_init = std::chrono::high_resolution_clock::now();

    //prepare spin lattice
    std::random_device rd;
    std::mt19937 gen(rd());

    //arrays for storing white and black sublattices
    int* whites = initSubLattice(subLen, gen);
    int* blacks = initSubLattice(subLen, gen);

    auto end_init = std::chrono::high_resolution_clock::now();
    auto duration_init = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);

    //run with timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int s{0}; s < nsweeps; s++){
        updateSubLattice(whites, blacks, size, subSize, temp, true, gen);
        updateSubLattice(blacks, whites, size, subSize, temp, false, gen);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    ofstream outFile("results.txt", ios::app);
    outFile << size << "\t" << temp << "\t" << nsweeps << "\t" << duration_init.count() << "\t" << duration.count() << endl;
    outFile.close();

    cout << "Size" << "\t" << "temp" << "\t" << "nsweeps" << "\t" << "Init" << "\t" << "Sim" << endl;
    cout << size << "\t" << temp << "\t" << nsweeps << "\t" << duration_init.count() << "\t" << duration.count() << endl;

    //cleanup
    delete[] whites;
    delete[] blacks;
    
    return 0;
}