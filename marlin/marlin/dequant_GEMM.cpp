#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void Batchvecquant3matmul_faster_cuda(torch::Tensor vec,
    torch::Tensor mat,
    torch::Tensor mul,
    torch::Tensor scales,
    torch::Tensor zeros
    );

void Batchvecquant3matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
){
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  Batchvecquant3matmul_faster_cuda(vec, mat, mul, scales, zeros);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Batchvecquant3matmul_faster", &Batchvecquant3matmul_faster, "Dequantize + GEMM");
}