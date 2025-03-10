/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdio>

int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

int milo_cuda_3bit(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

int milo_cuda_3bit_faster(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

int milo_cuda_3bit_with_zero(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
        void* zero,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = 64,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);


int milo_cuda_3bit_256_64(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

int milo_cuda_3bit_256_64_with_zero(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
        void* z,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

int milo_cuda_3bit_64_256_with_zero(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
        void* z,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void mul(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  int dev = A.get_device();
  int err = marlin_cuda(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}


void mul_3bit(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  int dev = A.get_device();
  int err;
  int pack = 0;
  int selection = 0;
  int thread_block_num = prob_k * prob_n / (sms  * 64 * 256) + 1;
  int global_reduce_64_256 = prob_k / (thread_block_num * 64) + 2;
  int global_reduce_256_64 = prob_k / (thread_block_num * 256) + 2;
  if ((global_reduce_256_64 < global_reduce_64_256) || (prob_n % 256 != 0))
    selection = 1;
  if (prob_k % 256 != 0)
    selection = 0;
  if (selection==1){
    err = milo_cuda_3bit_256_64(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    256,
    64,
    sms,
    max_par
  );
  }
  else{
    err = milo_cuda_3bit_faster(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    64,
    256,
    sms,
    max_par
  );
  }
  if (err == ERR_PROB_SHAPE) {
    //printf("groupsize : %d", groupsize);
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}

void mul_3bit_with_zero(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
  const torch::Tensor& z,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par){
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  }
  int dev = A.get_device();
  int err;
  int selection = 0;
  int thread_block_num = prob_k * prob_n / (sms  * 64 * 256) + 1;
  int global_reduce_64_256 = prob_k / (thread_block_num * 64) + 2;
  int global_reduce_256_64 = prob_k / (thread_block_num * 256) + 2;
  if ((global_reduce_256_64 < global_reduce_64_256) || (prob_n % 256 != 0))
    selection = 1;
  if (prob_k % 256 != 0)
    selection = 0;
  //int selection = (prob_k > prob_n * 3) ? 1 : 0;
  if (selection == 1){
    //printf("256 64 \n");
    err = milo_cuda_3bit_256_64_with_zero(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    z.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    256,
    64,
    sms,
    max_par
  );
  }
  else{
    //printf("64 256 \n");
    err = milo_cuda_3bit_64_256_with_zero(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    z.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    64,
    256,
    sms,
    max_par
  );
  }
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}


void mul_3bit_faster(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par){
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  }
    
  int dev = A.get_device();
  int err = milo_cuda_3bit_faster(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}


void mul_3bit_256_64(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par){
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  }
    
  int dev = A.get_device();
  int err = milo_cuda_3bit_256_64(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}

void mul_3bit_256_64_with_zero(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
  const torch::Tensor& z,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par){
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  }
    
  int dev = A.get_device();
  int err = milo_cuda_3bit_256_64_with_zero(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    z.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}


void mul_3bit_64_256_with_zero(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
  const torch::Tensor& z,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par){
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  }
    
  int dev = A.get_device();
  int err = milo_cuda_3bit_64_256_with_zero(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    z.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul", &mul, "Marlin FP16xINT4 matmul.");
  m.def("mul_3bit", &mul_3bit, "Marlin FP16xINT3 matmul.");
  m.def("mul_3bit_faster", &mul_3bit_faster, "Marlin FP16xINT3 matmul faster.");
  m.def("mul_3bit_with_zero", &mul_3bit_with_zero, "Marlin FP16xINT3 matmul with zeros.");
  m.def("mul_3bit_256_64", &mul_3bit_256_64, "Marlin FP16xINT3 matmul with tile 256x64.");
  m.def("mul_3bit_256_64_with_zero", &mul_3bit_256_64_with_zero, "Marlin FP16xINT3 matmul with zeros on tile 256x64.");
  m.def("mul_3bit_64_256_with_zero", &mul_3bit_64_256_with_zero, "Marlin FP16xINT3 matmul with zeros on tile 64x256.");
}

