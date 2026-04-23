#include "ising.hpp"
#include <fstream>

int main(int argc, char* argv[]){

    if(argc != 4 && argc != 5){
        cerr << "Usage: " << argv[0] << " <grid_size=16> <temperature=1.0> <sweeps=1000> [pbc=0|1]" << endl;
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
    bool pbc = (argc == 5) ? (stoi(argv[4]) != 0) : false;

    //prepare spin lattice
    std::random_device rd;
    std::mt19937 gen(rd());

    Grid grid = initGrid(size, gen);

    //run simulation
    //cout << "Initial lattice:" << endl;
    //printLattice(grid);

    //open output file
    ofstream outFile("magnetization.txt");
    if (!outFile) {
        cerr << "Error opening output file." << endl;
        return -1;
    }
    outFile << "Sweeps" << "\t" << "Mag" << "\t" << "Temp" << "\t" << "W_flips" << "\t" << "B_flips" << endl;
  
    cout << "Running " << sweeps << " Monte Carlo sweeps at temperature " << temp
         << (pbc ? " with periodic boundary conditions" : " with open boundary conditions")
         << ":" << endl;
    
    for(int s{0}; s < sweeps; s++){
        int whiteFlips{0};
        int blackFlips{0};
        whiteFlips = updateSublattice(grid.lattice, grid.size, temp, gen, true, pbc);
        blackFlips = updateSublattice(grid.lattice, grid.size, temp, gen, false, pbc);
        writeSweepToFile(outFile, grid, temp, s, whiteFlips, blackFlips);
    }
    
    cout << "Final lattice:" << endl;
    printLattice(grid);

    return 0;
}