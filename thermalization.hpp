#ifndef __THERMALIZATION_HPP__
#define __THERMALIZATION_HPP__

#include "ising_shared.hpp"

#define CRIT_TEMP 2.26918531421


/* annealing */

template<BoundaryCondition BC>
void anneal(
    DataBlockShared *d, double endTemp, double startTemp = 5.0, int annealSweeps = 5000)
{
    fmt::println("Annealing phase (annealing T: {} -> {})", startTemp, endTemp);
    double beta_start = 1.0 / startTemp;
    double beta_end = 1.0 / endTemp;
    int sweep = 0;
    KernelConfig config = kernelConfig(d->size);
    int n_TilesRow = d->size / __TILE_DIM_X__;
    while(sweep < annealSweeps){
        double beta = beta_start + (beta_end - beta_start) * (static_cast<double>(sweep) / annealSweeps);
        d->temp = 1.0 / beta;
        //exptable
        float expTable[9];
        for(int i{0}; i<9; i++){
            expTable[i] = exp(-(float)_deltaE[i] * beta);
        }
        cudaMemcpyToSymbol(d_expTable, expTable, 9 * sizeof(float));
        updateTileKernel<BC><<<config.grid, config.block, config.shared_mem_bytes>>>(
            d->lattice,
            d->size,
            0,
            n_TilesRow,
            d->state
        );
        updateTileKernel<BC><<<config.grid, config.block, config.shared_mem_bytes>>>(
            d->lattice, 
            d->size, 
            1,
            n_TilesRow,
            d->state
        );
        sweep++;
    }
}

/* thermalization */

#endif // __THERMALIZATION_HPP__