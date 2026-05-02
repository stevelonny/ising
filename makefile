run: run.cu ising_gpu.hpp
	nvcc -o run.x run.cu
bench_gpu: bench_gpu.cu ising_gpu.hpp
	nvcc -o bench_gpu.x bench_gpu.cu
bench_cpu: bench_cpu.cpp ising_cpu.hpp
	g++ -o bench_cpu.x bench_cpu.cpp
testcuda: testcuda.cu isingcuda.hpp
	nvcc -o testcuda.x testcuda.cu
clean:
	rm -f *.x *.o