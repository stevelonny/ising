// ising 2d lattice using 1darray for storing spins in a chekcerboard manner
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>

using namespace std;

int inline getBlackIndex(int i, int j, int subSize){
    return i * subSize + ((i % 2 == 0) ? (j - 1) / 2 : j / 2);
}

int inline getWhiteIndex(int i, int j, int subSize){
    return i * subSize + ((i % 2 == 0) ? j / 2 : (j - 1) / 2);
}

//maybe better to have separate func for even and odd rows
int getNeighboursSum(int* opponent, int index, int size, int subSize, bool isWhite){
    int row = index / subSize;
    int col = index % subSize;
    bool rowParity = (row % 2 == 0);
    if(isWhite) rowParity = !rowParity; //white and black sublattice have opposite parity
    /* nn indices */
    int up      = index - subSize;
    int down    = index + subSize;
    // left and right index switch places for each change of row's parity ("adjacent")
    //int adj1    = index;
    //int adj2    = index + (rowParity ? 1 : -1);

    int sum{0};
    
    if (up >= 0)            sum += opponent[up];
    if (down < size*size/2) sum += opponent[down];
    if (rowParity){ //even row
        sum += opponent[index];                     //right
        if(col > 0) sum += opponent[index - 1];     //left
    }
    else{ //odd row
        sum += opponent[index];                             //left
        if(col < subSize - 1) sum += opponent[index + 1];   //right
    }

    return sum;

}

double uniform01(std::mt19937& gen){
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

int updateSubLattice(int* subLattice, int* opponent, int size, int subSize, double temp, bool isWhite, std::mt19937& gen){
    int subLen = size*subSize;
    int flips{0};
    // double for cycle for odd/even row?
    for(int i{0}; i < subLen; i++){
        int nearSum = getNeighboursSum(opponent, i, size, subSize, isWhite);
        double deltaE = 2.*subLattice[i] * nearSum;
        if(uniform01(gen) < exp(-deltaE / temp)){
            subLattice[i] *= -1;
            flips++;
        }
    }
    return flips;
}

void printLattice(int* whites, int* blacks, int size, int subSize){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if((i + j) % 2 == 0){
                cout << (whites[getWhiteIndex(i, j, subSize)]>0 ? "1" : "0") << " ";
            } else{
                cout << (blacks[getBlackIndex(i, j, subSize)]>0 ? "1" : "0") << " ";
            }
        }
        cout << endl;
    }
}

int* initSubLattice(int subLen, std::mt19937& gen){
    int* subLattice = new int[subLen];
    std::bernoulli_distribution dist(0.5);
    for (int i = 0; i < subLen; i++) {
        subLattice[i] = dist(gen) ? 1 : -1;
    }
    return subLattice;
}

int getMagnetization(int* subLattice, int subLen){
    int mag = 0;
    for(int i = 0; i < subLen; i++){
        mag += subLattice[i];
    }
    return mag;
}