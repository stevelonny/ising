
INCLUDE_FLAGS = -I./fmt/include -arch=sm_75
LIB_FLAGS = -L./fmt/build -lfmt -lcudadevrt

#run: run.cu ising_gpu.hpp fmt/src/format.cc
#	nvcc $(INCLUDE_FLAGS) $(LIB_FLAGS) -o run.x run.cu -diag-suppress=2417,128
run_hotstart.x: run_hotstart.cu ising_gpu.hpp ising_shared.hpp fmt/build/libfmt.a
	nvcc -O3 $(INCLUDE_FLAGS) -o run_hotstart.x run_hotstart.cu $(LIB_FLAGS) -diag-suppress=2417,128
run_shared.x: run_shared.cu ising_gpu.hpp ising_shared.hpp fmt/build/libfmt.a
	nvcc $(INCLUDE_FLAGS) -rdc=true -o run_shared.x run_shared.cu $(LIB_FLAGS) -lcudadevrt -diag-suppress=2417,128
run.x: run.cu ising_gpu.hpp ising_cpu.hpp fmt/build/libfmt.a
	nvcc $(INCLUDE_FLAGS) -o run.x run.cu $(LIB_FLAGS) -diag-suppress=2417,128

fmt/build/libfmt.a:
	cd fmt && mkdir -p build && cd build && cmake .. && make && cd ../..

bench_gpu_operation.x: bench_gpu_operation.cu ising_gpu.hpp fmt/build/libfmt.a
	nvcc $(INCLUDE_FLAGS) -o bench_gpu_operation.x bench_gpu_operation.cu $(LIB_FLAGS) -diag-suppress=2417,128

bench_gpu: bench_gpu.cu ising_gpu.hpp
	nvcc $(INCLUDE_FLAGS) -o bench_gpu.x bench_gpu.cu $(LIB_FLAGS) -diag-suppress=2417,128
bench_comparison: bench_comparison.cu ising_gpu.hpp ising_shared.hpp
	nvcc $(INCLUDE_FLAGS) -o bench_comparison.x bench_comparison.cu $(LIB_FLAGS) -diag-suppress=2417,128
bench_cpu: bench_cpu.cpp ising_cpu.hpp
	g++ -o bench_cpu.x bench_cpu.cpp
testcuda: testcuda.cu ising_gpu.hpp ising_shared.hpp fmt/build/libfmt.a
	nvcc $(INCLUDE_FLAGS) -o testcuda.x testcuda.cu $(LIB_FLAGS) -diag-suppress=2417,128
clean:
	rm -f *.x *.o
clean-fmt:
	cd fmt/build && make clean