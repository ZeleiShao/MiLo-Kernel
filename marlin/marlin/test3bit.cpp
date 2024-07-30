#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include<iostream>
//目的是观察torch中type为half的数据变为void*后数据读取的变化


int main(){
        int m = 4;
        int k = 4;
        auto A = torch.ones((m, k), dtype=torch.half);
        auto A1 = A.data_ptr();

        const int* A1_ptr = (const int*) A1;

        std::cout << A[0] << " " << A1_ptr[0] << " " std::endl;
        return 0;

}










