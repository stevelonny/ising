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

int getMagnetization(int* blacks, int* whites, int subLen){
    int mag = 0;
    for(int i = 0; i < subLen; i++){
        mag += blacks[i];
        mag += whites[i];
    }
    return mag;
}

int main(int argc, char* argv[]){

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

    int subSize = size/2; //numbers of columns for each sublattice
    int subLen = size*subSize; //length of each sublattice array (total number of elements)

    //prepare spin lattice
    std::random_device rd;
    std::mt19937 gen(rd());

    // arrays for storing white and black sublattices
    int* whites = initSubLattice(subLen, gen);
    int* blacks = initSubLattice(subLen, gen);

    ofstream outFile("magnetization.txt");
    if (!outFile) {
        cerr << "Error opening output file." << endl;
        return -1;
    }
    outFile << "Sweeps" << "\t" << "Mag" << "\t" << "Temp" << "\t" << "W_flips" << "\t" << "B_flips" << endl;
  

    //printLattice(whites, blacks, size, subSize);
    for(int s{0}; s < sweeps; s++){
        int whiteFlips = updateSubLattice(whites, blacks, size, subSize, temp, gen);
        int blackFlips = updateSubLattice(blacks, whites, size, subSize, temp, gen);
        int magnetization = getMagnetization(blacks, whites, subLen);
        outFile << s << "\t" << magnetization << "\t" << temp << "\t" << whiteFlips << "\t" << blackFlips << endl;
    }

    //printLattice(whites, blacks, size, subSize);

    delete[] whites;
    delete[] blacks;

    return 0;
}