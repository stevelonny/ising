# Ising 2D Simulation

Simple code to simulate an Ising 2D lattice with a Metropolis algorithm on NVIDIA GPUs. Project made for the Metodi Computazionali per la Fisica @ UNIMI A.Y. 2025-2026. The main goal is to explore different GPU implementations and compare performance/accuracy.

Key ideas implemented here:
- CPU reference implementation (`ising_cpu.hpp`).
- GPU implementations with increasing memory locality and synchronization sophistication:
	- simple per-sublattice global-memory kernels (`ising_gpu.hpp` + `run.cu`),
	- shared-memory tiled kernels (`ising_shared.hpp` + `run_shared.cu`),
	- cooperative-grid launch tiled kernels (cooperative groups, `ising_shared.hpp`).

## Quick setup
Depends on: CUDA toolkit (nvcc), a compatible NVIDIA GPU, and `fmt` (the repository contains a local `fmt/` directory used by the makefile).

Clone this repo recursively to include `fmt` dependencies.

```bash
git clone --recursive https://github.com/stevelonny/ising.git
```

Build fmt static library.

```bash
cd fmt && mkdir -p build && cd build
cmake .. && make
cd ../..
```

Build the binaries (examples):

```bash
make run.x              # builds run.x
make run_shared.x       # builds run_shared.x
make run_hotstart.x     # builds run_hotstart.x
```

## Main source files

### Api
- [ising_cpu.hpp](ising_cpu.hpp): CPU reference implementation (Metropolis sweeps on white/black sublattices). Useful for validation and debugging.
- [ising_gpu.hpp](ising_gpu.hpp): "global" GPU implementation. Uses two device arrays (`whites`/`blacks`) storing checkerboard sublattices, per-thread CURAND states and kernels:
	- `initCurand`, `initSublatticeKernel` — initialize RNG and spins on device.
	- `updateSublatticeKernel` — performs Metropolis updates reading neighbor spins via index arithmetic (global memory reads).
	- `reduceMagnetizationThrust` / `reduceMagnetizationKernel` — compute magnetization (Thrust-based or simple reduction kernel).
	- `blockSweepKernel` — performs alternating white/black updates from host by launching `updateSublatticeKernel` twice per sweep.

- [ising_shared.hpp](ising_shared.hpp): shared-memory tiled implementation. Main ideas:
	- Tiles (blocks) load a tile plus halo into shared memory (`s_tile_raw`), so neighbor reads are mostly from fast shared memory.
	- Each block updates both white and black sites inside the tile and writes back to global memory.
	- Per-tile reduction in shared memory yields partial magnetization contributions which are atomically accumulated to a per-sweep counter.
	- Implements both a normal tiled kernel (`updateTileKernel`) and a cooperative-grid version (`cooperativeUpdateTileKernel`) which uses `cooperative_groups` and `cudaLaunchCooperativeKernel`.

### Implementation examples
- [run.cu](run.cu): host program for the global implementation (uses `ising_gpu.hpp`). Typical flow:
	- Parse arguments: `<power2> <T> <nblocksweeps> <blocksize> <obc|pbc> [run_tag]`.
	- Initialize device arrays and CURAND states.
	- Optional annealing, then thermalization loops.
	- For measurement it uses `blockSweepKernel` to perform block sweeps and writes summary statistics to a `.dat` file.

- [run_shared.cu](run_shared.cu): host program for launching experiments (uses `ising_shared.hpp`). Notes:
	- Parses arguments: `./run_shared.x -L <log2_size> -T <temp> -N <nsweeps> -b <obc|pbc> [-t <tag>] [-o <output.csv>] [-s <seed>] [-v]`.
	- Uses `updateTileKernel` with shared memory
    - Simulation:
        - Anneal phase of 1000 sweeps
        - Thermalization phase checks half windows of 10000 sweeps for convergence of magnetization (stopping when the difference between the two halves is less than `0.05 * sqrt(var1 + var2)`).
		- Measurement phase of `nsweeps` sweeps, where it collects magnetization data.
    - Integrates datablocking algorithms for computing statistical errors. Appends a summary to `master_results.csv`

- [run_hotstart.cu](run_hotstart.cu): host program for launching experiments without annealing or convergence checks:
	- Performs a long thermalization.
	- Stores raw magnetization sweeps to a `.dat` file.

- `bench_*` (e.g. `bench_gpu.cu`, `bench_gpu_operation.cu`, `bench_ising.cu`): small benchmarks and operation-level timing helpers.
- `testcuda.cu`: program to test correctness of the various implementations.

## Additional utilities
This project includes several helper scripts and notebooks useful for benchmarking, running large ensembles, and post-processing results. Brief summaries and usage notes are below.

- `run_benchmarks.py`: Python benchmark driver that compiles a small CUDA comparison (`bench_comparison.cu`) and runs a configurable grid of experiments (implementations: `global`, `shared`, `shared_cooperative`) across lattice sizes. It:
	- Verifies `nvcc` and `nvidia-smi`, compiles the benchmark, runs experiments, aggregates results and saves CSV/JSON outputs under `bench_results/` and a plot `comparison.png`.
	- Requires Python packages listed in `requirements.txt` (pandas, numpy, matplotlib, seaborn, etc.). Use `setup_venv.sh` to create a virtual environment and install dependencies.
	- Usage: `python run_benchmarks.py` (see script header for environment variables `BENCH_MAX_SIZE`, `BENCH_REPEATS`).

- `peak.ipynb`: Jupyter notebook for analysis of `master_results.csv`. It loads per-run aggregated observables, computes susceptibility and Binder cumulant, pools errors with a simple blocked-error routine and plots:
	- Magnetization, susceptibility, Binder cumulant and integrated autocorrelation estimates as a function of temperature and lattice size.
	- Open the notebook in JupyterLab/Notebook and run cells; it expects the `master_results.csv` file produced by `run_shared.x`.

- `runtime.ipynb`: Notebook for quick runtime comparisons between CPU and GPU binaries. It contains plotting cells that expect `results.txt` and `results_gpu.txt` (simple tab-separated summaries produced by bench programs). Useful to visualize initialization and simulation time scaling with lattice size.

- `setup_venv.sh`: Creates a Python virtual environment at `.venv` and installs packages from `requirements.txt`. Recommended before running `run_benchmarks.py`:
	```bash
	./setup_venv.sh
	source .venv/bin/activate
	python run_benchmarks.py
	```

- `run_all_shared.sh`: Convenience script to run long ensemble sweeps using the `run_shared.x` binary across multiple temperatures and lattice sizes. Tunable variables at the top:
	- `NSWEEPS` (sweeps per run), `POWERS` (array of power-of-two exponents), `TEMP_LOW`, `TEMP_HIGH`, `TEMP_STEP`, `NUM_RUNS`.
	- Build `run_shared.x` first, then run this script to produce many measurement files (one per run/tag).

- `run_all_temps.sh`: Wrapper to run `run.x` across a range of temperatures and both `pbc`/`obc` boundary conditions, with logging, retry logic and a summary file. Usage example:
	```bash
	./run_all_temps.sh 10 10 100 3
	# -> power2=10, nblocksweeps=10, blocksize=100, num_runs=3
	```

## Differences between the GPU approaches

1) Global (direct global-memory kernels)
	 - Each thread updates a single element of a sublattice and reads its neighbors via computed global indices.
	 - Simple to implement and portable across GPUs.
	 - Pros: few constraints on launch configuration, straightforward CURAND usage per thread.
	 - Cons: high global-memory traffic for neighbor reads; less reuse of cached data, potentially lower throughput for large lattices.

2) Shared-memory tiled kernels
	 - Each block loads an entire tile (plus halo) into shared memory, then performs updates reading from shared memory where possible.
	 - Pros: much lower global-memory traffic for neighbor reads, better data locality and coalescing, often faster for medium-to-large tiles.
	 - Cons: more complex indexing, requires tuning tile/block sizes and careful shared-memory sizing. Reduction of magnetization typically performed per tile then atomically accumulated.

3) Cooperative-grid tiled kernel (cooperative launch)
	 - Uses CUDA cooperative groups and `cudaLaunchCooperativeKernel` so the kernel can perform multiple sweep iterations and call `grid.sync()` to synchronize across the entire grid from device code.
	 - Pros: reduces host<->device kernel launch overhead (many sweeps can be performed inside one kernel) and can express global-level synchronization without returning to host.
	 - Cons: requires GPU and driver support for cooperative launches; occupancy must be carefully managed (the kernel must fit the device's limits), and some platforms may not allow cooperative launches or may limit grid size.
	 - Refer to [Cooperative launch of tile updates #3](https://github.com/stevelonny/ising/pull/3)

## Notes, caveats and tips
- The code assumes power-of-two lattice sizes; periodic boundary optimizations use bitwise wrapping.
- Cooperative kernels require device support (compute capability sm_70).
- RNG: CURAND Philox (`curandStatePhilox4_32_10_t`) is used.

## License
This project is licensed under the [MIT License](LICENSE).
