# Ising 2D Simulation
Simple code to simulate an Ising 2D lattice with a basic algorithm that can be run both on CPU and GPU. Project made for the Metodi Computazionali per la Fisica @ UNIMI A.Y. 2025-2026. The scope of this project is to understand coding on CUDA hardware.

## Files
- `ising_cpu.hpp`
- `ising_gpu.hpp`
- `testcuda.cu`
- `bench_cpu.cpp`
- `bench_gpu.cu`

## ToDo
- [x] Test indexing
- [x] Test coherent results from cpu vs gpu
- [x] Use multiple size of nthreads or nblocks, and multiple lattice size
- [ ] Input settings
- [ ] Observables/Output (magnetization...)
- [ ] Notebook for analysis
- [ ] Comparison for observables in different situations
- [ ] Border conditions: obc, pbc... halo?
- [ ] ...



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
