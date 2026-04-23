#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

using namespace std;

#define _TEMP 1.0

struct Grid {
    int size;
    int **lattice;
};

bool isWhite(int i, int j){
    return (i + j) % 2 == 0;
}

Grid initGrid(int size, std::mt19937& gen){
    
    std::bernoulli_distribution d(0.5);
    
    Grid grid;
    grid.size = size;

    grid.lattice = new int*[size];
    for (int i = 0; i < size; i++){
        grid.lattice[i] = new int[size];
        for (int j = 0; j < size; j++){
            grid.lattice[i][j] = d(gen) ? 1 : -1;
        }
    }
    return grid;
}


int getNeighbourSum(int* const* lattice, int size, int i, int j, bool pbc = false){
    if(!pbc){
        int sum{0};
        int i_up = i - 1;
        int i_down = i + 1;
        int j_left = j - 1;
        int j_right = j + 1;

        if (i_up >= 0) sum += lattice[i_up][j];
        if (i_down < size) sum += lattice[i_down][j];
        if (j_left >= 0) sum += lattice[i][j_left];
        if (j_right < size) sum += lattice[i][j_right];
        return sum;
    }
    int sum{0};
    int i_up = (i - 1 + size) % size;
    int i_down = (i + 1) % size;
    int j_left = (j - 1 + size) % size;
    int j_right = (j + 1) % size;
    
    sum += lattice[i_up][j];
    sum += lattice[i_down][j];
    sum += lattice[i][j_left];
    sum += lattice[i][j_right];
    return sum;
}


double getMagnetization(const Grid& grid){
    int mag{0};
    for (int i = 0; i < grid.size; i++){
        for (int j = 0; j < grid.size; j++){
            mag += grid.lattice[i][j];
        }
    }
    return static_cast<double>(mag);
}

void printLattice(const Grid& grid){
    for (int i = 0; i < grid.size; i++){
        for (int j = 0; j < grid.size; j++){
            cout << (grid.lattice[i][j] > 0 ? "1" : "0") << " ";
        }
        cout << endl;
    }
}

void printNeighbourSums(const Grid& grid){
    for (int i = 0; i < grid.size; i++){
        for (int j = 0; j < grid.size; j++){
            cout << getNeighbourSum(grid.lattice, grid.size, i, j) << " ";
        }
        cout << endl;
    }
}

double uniform01(std::mt19937& gen){
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

//update white sublattice
int updateSublattice(int** lattice, int size, double temp, std::mt19937& gen, bool isWhiteSublattice = true, bool pbc = false){
    int flips{0};
    for(int i{0}; i < size; i++){
        int offset = (isWhiteSublattice ? (i % 2) : (1 - i % 2));
        for(int j{0}; j < size / 2; j++){
            int fullJ = 2*j + offset;
            int neighbourSum = getNeighbourSum(lattice, size, i, fullJ, pbc);
            double deltaE = 2 * lattice[i][fullJ] * neighbourSum;
            if (deltaE <= 0 || exp(-deltaE / temp) > uniform01(gen)) {
                lattice[i][fullJ] *= -1;
                flips++;
            }
        }
    }
    return flips;
}

//single monte carlo sweep
void monteCarloSweep(Grid& grid, double temp, std::mt19937& gen, bool pbc){
    updateSublattice(grid.lattice, grid.size, temp, gen, true, pbc);
    updateSublattice(grid.lattice, grid.size, temp, gen, false, pbc);
}

void writeSweepToFile(ofstream& outFile, const Grid& grid, double temp, int sweep, int whiteFlips, int blackFlips){
    double mag = getMagnetization(grid);
    outFile << sweep << "\t" << mag << "\t" << temp << "\t" << whiteFlips << "\t" << blackFlips << endl;
}