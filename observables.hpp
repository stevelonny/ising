#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fmt/format.h>
#include <fmt/ostream.h>

struct BlockResult {
    double mean;
    double error;
    double tau_int;
    std::vector<int> n_blocks_per_level;
    std::vector<double> mean_per_level;
    std::vector<double> error_per_level;
    std::vector<double> variance_per_level;
};

BlockResult calculateBlockedError(const std::vector<double>& data, int max_levels = 15){
    std::vector<double> current_data = data;
    
    double mean{0.};
    for(double val : data) mean += val;
    mean /= data.size();

    double max_err{0.};
    double naive_err{0.};

    std::vector<int> n_blocks_per_level;
    std::vector<double> mean_per_level;
    std::vector<double> error_per_level;
    std::vector<double> variance_per_level;

    for (int level{0}; level<max_levels; ++level) {
        int n = current_data.size();
        if (n<2) break;

        double var{0.};
        double current_mean{0.};
        for(double v : current_data) current_mean += v;
        current_mean /= n;

        for(double v : current_data) var += (v - current_mean) * (v - current_mean);
        var /= (n - 1); 

        double err = std::sqrt(var / n);
        
        if (level == 0) naive_err = err;

        if (err > max_err) max_err = err; 
        
        n_blocks_per_level.push_back(n);
        mean_per_level.push_back(current_mean);
        error_per_level.push_back(err);
        variance_per_level.push_back(var);
        
        int limit = (n / 2) * 2;
        std::vector<double> next_data(limit / 2);
        for (int i{0}; i<limit; i+=2) {
            next_data[i/2] = 0.5 * (current_data[i] + current_data[i+1]);
        }
        current_data = next_data;
    }
    
    double tau = 0.0;
    if (naive_err > 0.0) {
        tau = 0.5 * ((max_err * max_err) / (naive_err * naive_err) - 1.0);
    }
    if (tau < 0.0) tau = 0.0;

    return {mean, max_err, tau, n_blocks_per_level, mean_per_level, error_per_level, variance_per_level};
}

/* print functions */
void printBlockResult(const BlockResult& result, std::ostream& fileout, const std::string& obs_name){
    // obs_name, lv, n_blocks, mean, error, variance
    for(int i{0}; i<result.n_blocks_per_level.size(); i++){
        fmt::println(
            fileout,
            "{:<10}{:<3}{:<10}{:<15.8e}{:<15.8e}{:<15.8e}",
            obs_name,
            i,
            result.n_blocks_per_level[i],
            result.mean_per_level[i],
            result.error_per_level[i],
            result.variance_per_level[i]
        );
    }
}