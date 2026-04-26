// ising 2d lattice using 1darray for storing spins in a chekcerboard manner
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

int inline getBlackIndex(int i, int j, int subSize){
    return i * subSize + ((i % 2 == 0) ? (j - 1) / 2 : j / 2);
}

int inline getWhiteIndex(int i, int j, int subSize){
    return i * subSize + ((i % 2 == 0) ? j / 2 : (j - 1) / 2);
}

//maybe better to have separate func for even and odd rows
int getNeighboursSum(int* opponent, int index, int size, int subSize){
    int row = index / subSize;
    int col = index % subSize;
    bool rowParity = (row % 2 == 0);
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

int getWhiteNeighboursSum(int* blacks, int index, int size, int subSize){
    int row = index / subSize;
    int col = index % subSize;
    int fullJ = 2*col + (row%2);

    int sum{0};
    if (row > 0)        sum += blacks[getBlackIndex(row - 1, fullJ, subSize)];
    if (row < size-1)   sum += blacks[getBlackIndex(row + 1, fullJ, subSize)];
    if (fullJ > 0)      sum += blacks[getBlackIndex(row, fullJ - 1, subSize)];
    if (fullJ < size-1) sum += blacks[getBlackIndex(row, fullJ + 1, subSize)];

    return sum;
}

int getBlackNeighboursSum(int* whites, int index, int size, int subSize){
    int row = index / subSize;
    int col = index % subSize;
    int fullJ = 2*col + ((row + 1) % 2);

    int sum{0};
    if (row > 0)        sum += whites[getWhiteIndex(row - 1, fullJ, subSize)];
    if (row < size-1)   sum += whites[getWhiteIndex(row + 1, fullJ, subSize)];
    if (fullJ > 0)      sum += whites[getWhiteIndex(row, fullJ - 1, subSize)];
    if (fullJ < size-1) sum += whites[getWhiteIndex(row, fullJ + 1, subSize)];

    return sum;
}

double uniform01(std::mt19937& gen){
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

int updateSubLattice(int* subLattice, int* opponent, int size, int subSize, double temp, std::mt19937& gen){
    int subLen = size*subSize;
    int flips{0};
    // double for cycle for odd/even row?
    for(int i{0}; i < subLen; i++){
        int nearSum = getNeighboursSum(opponent, i, size, subSize);
        double deltaE = 2.*subLattice[i] * nearSum;
        if(deltaE < 0 || uniform01(gen) < exp(-deltaE / temp)){
            subLattice[i] *= -1;
            flips++;
        }
    }
    return flips;
}

int updateWhitesubLattice(int* whites, int* blacks, int size, int subSize, double temp, std::mt19937& gen){
    int subLen = size*subSize;
    int flips{0};
    // double for cycle for odd/even row?
    for(int i{0}; i < subLen; i++){
        int nearSum = getNeighboursSum(blacks, i, size, subSize);
        double deltaE = 2.*whites[i] * nearSum;
        if(deltaE < 0 || uniform01(gen) < exp(-deltaE / temp)){
            whites[i] *= -1;
            flips++;
        }
    }
    return flips;
}

int updateBlacksubLattice(int* blacks, int* whites, int size, int subSize, double temp, std::mt19937& gen){
    int subLen = size*subSize;
    int flips{0};
    // double for cycle for odd/even row?
    for(int i{0}; i < subLen; i++){
        int nearSum = getNeighboursSum(whites, i, size, subSize);
        double deltaE = 2.*blacks[i] * nearSum;
        if(deltaE < 0 || uniform01(gen) < exp(-deltaE / temp)){
            blacks[i] *= -1;
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

int main(){

    // arrays for storing white and black sublattices
    int* whites;
    int* blacks;

    int size{8}; //size of the total lattice
    int subSize{size/2}; //numbers of columns for each sublattice
    int subLen{size*subSize}; //length of each sublattice array (total number of elements)

    int sweeps{100};
    double temp{1.0};

    //prepare spin lattice
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.5);

    whites = new int[subLen];
    blacks = new int[subLen];
    for (int i = 0; i < subLen; i++) {
        //whites[i] = 1;
        //blacks[i] = -1;
        whites[i] = dist(gen) ? 1 : -1;
        blacks[i] = dist(gen) ? 1 : -1;
    }

    //printLattice(whites, blacks, size, subSize);
    for(int s{0}; s < sweeps; s++){
        int whiteFlips = updateSubLattice(whites, blacks, size, subSize, temp, gen);
        int blackFlips = updateSubLattice(blacks, whites, size, subSize, temp, gen);
        cout << "sweep " << s << ": white=" << whiteFlips << " black=" << blackFlips << endl;
    }

    printLattice(whites, blacks, size, subSize);

}