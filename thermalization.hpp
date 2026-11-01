#ifndef __THERMALIZATION_HPP__
#define __THERMALIZATION_HPP__

#include "ising_shared.hpp"
#include "observables.hpp"

#define CRIT_TEMP 2.26918531421
#define MAX_THERMALIZATION_SWEEPS 16384

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
        double beta = beta_start + (beta_end - beta_start) * (static_cast<double>(sweep) / static_cast<double>(annealSweeps-1));
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

struct ThermalizationResult{
    bool thermalized;
    int sweeps;
};

bool converged(
    const BlockResult& prevResult, const BlockResult& currResult,
    double relativeTolerance = 0.05, double sigmaTolerance = 2.0
){
    double diff{std::abs(currResult.mean - prevResult.mean)};
    double unc{std::sqrt(currResult.error * currResult.error + prevResult.error * prevResult.error)};
    double scale{std::max(
        std::abs(currResult.mean),
        std::abs(prevResult.mean),
        1.0)
    };
    double relDiff{diff/scale};

    return (relDiff < relativeTolerance) || (diff < sigmaTolerance * unc); //use `||` to require only one
}

template<BoundaryCondition BC>
ThermalizationResult thermalize(
    DataBlockShared *d,
    int windowSize = 2048,
    int maxSweeps = MAX_THERMALIZATION_SWEEPS,
    int requiredConsecutiveWindows = 3
){
    fmt::println("Thermalization phase (T = {})", d->temp);
    int consecutiveWindows{0};
    bool havePreviousWindow{false};
    BlockResult prevAbsMagResult;
    BlockResult prevM2Result;
    BlockResult prevEnergyResult;
    int maxWindows {maxSweeps/windowSize};
    int nWindows{0};
    int maxLevels{8};
    //init mag and energy
    cudaMalloc(d->magnetization, windowSize * sizeof(int));
    cudaMalloc(d->energy, windowSize * sizeof(int));

    int* magnetization = new int[windowSize];
    int* energy = new int[windowSize];
    std::vector<double> absMag_data(windowSize);
    std::vector<double> m2_data(windowSize);
    std::vector<double> energy_data(windowSize);

    while(consecutiveWindows < requiredConsecutiveWindows && nWindows < maxWindows){
        cudaMemset(d->magnetization, 0, windowSize * sizeof(int));
        cudaMemset(d->energy, 0, windowSize * sizeof(int));
        //update
        blockSweepKernelShared<BC>(d, windowSize);
        nWindows++;
        //compute magnetization and energy
        checkCuda(
            cudaMemcpy(magnetization, d->magnetization, windowSize * sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy magnetization from device to host"
        );
        checkCuda(
            cudaMemcpy(energy, d->energy, windowSize * sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy energy from device to host"
        );
        for(int i{0}; i<windowSize; i++){
            absMag_data[i] = std::abs(static_cast<double>(magnetization[i]));
            m2_data[i] = static_cast<double>(magnetization[i]) * static_cast<double>(magnetization[i]);
            energy_data[i] = static_cast<double>(energy[i]);
        }
        BlockResult absMag = calculateBlockedError(absMag_data, maxLevels);
        BlockResult m2 = calculateBlockedError(m2_data, maxLevels);
        BlockResult energyRes = calculateBlockedError(energy_data, maxLevels);
        if(havePreviousWindow){
            bool absMagConverged = converged(prevAbsMagResult, absMag);
            bool m2Converged = converged(prevM2Result, m2);
            bool energyConverged = converged(prevEnergyResult, energyRes);
            if(absMagConverged && m2Converged && energyConverged){
                consecutiveWindows++;
            }
            else{
                consecutiveWindows = 0;
            }
        }
        prevAbsMagResult = absMag;
        prevM2Result = m2;
        prevEnergyResult = energyRes;
        havePreviousWindow = true;
    }
    ThermalizationResult result;
    result.thermalized = (consecutiveWindows >= requiredConsecutiveWindows);
    result.sweeps = nWindows * windowSize;
    delete[] magnetization;
    delete[] energy;
    cudaFree(d->magnetization);
    cudaFree(d->energy);
    d->magnetization = nullptr;
    d->energy = nullptr;
    return result;
}

#endif // __THERMALIZATION_HPP__