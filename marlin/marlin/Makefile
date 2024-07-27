CC = g++
CFLAGS = -std=c++14

NVCC = nvcc
NVCC_FLAGS = --gpu-architecture=sm_80 -std=c++14 -O3 -Wno-deprecated-gpu-targets

LIBRARIES = -L/${CUDA_DIR}/lib64 -lcudart -lm

marlin: main.o marlin_cuda_kernel.o marlin_3bit_cuda_kernel.o 
	$(CC) $^ -o $@ $(LIBRARIES)

main.o: marlin_cuda.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

marlin_cuda_kernel.o: marlin_cuda_kernel.cu
	$(CC) $(CFLAGS) -c $^ -o $@

marlin_3bit_cuda_kernel.o: marlin_3bit_cuda_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o marlin