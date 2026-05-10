
INCLUDE_FLAGS = -I./fmt/include
LIB_FLAGS = -L./fmt/build -lfmt

#run: run.cu ising_gpu.hpp fmt/src/format.cc
#	nvcc $(INCLUDE_FLAGS) $(LIB_FLAGS) -o run.x run.cu -diag-suppress=2417,128
run.x: run.cu ising_gpu.hpp ising_cpu.hpp fmt/build/libfmt.a
	nvcc $(INCLUDE_FLAGS) -o run.x run.cu $(LIB_FLAGS) -diag-suppress=2417,128

fmt/build/libfmt.a:
	cd fmt && mkdir -p build && cd build && cmake .. && make && cd ../..

bench_gpu: bench_gpu.cu ising_gpu.hpp
	nvcc -o bench_gpu.x bench_gpu.cu
bench_cpu: bench_cpu.cpp ising_cpu.hpp
	g++ -o bench_cpu.x bench_cpu.cpp
testcuda: testcuda.cu isingcuda.hpp
	nvcc -o testcuda.x testcuda.cu
clean:
	rm -f *.x *.o
clean-fmt:
	cd fmt/build && make clean