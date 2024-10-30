#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};
using FragB = Vec<half2, 2>;
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

__global__ void BatchVecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  half2* __restrict__ scales,
    const  half2* __restrict__ zeros,
    int height,
    int width,
    int batchsize = 1
);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;

void Batchvecquant3matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  BatchVecQuant3MatMulKernelFaster<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    (half2*)scales.data_ptr(),
    (half2*)zeros.data_ptr(),
    height, width, batch
  );
}



__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline FragB dequant_faster(unsigned int q) {
  const int LO = 0x00070007;
  const int HI = 0x00380038;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-4` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64006400;
  const int MUL = 0x30003000;
  const int ADD = 0xd800d800;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

__global__ void BatchVecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  half2* __restrict__ scales,
    const  half2* __restrict__ zeros,
    int height,
    int width,
    int batchsize
) {
    const int BLOCKWIDTH  = 256;
    const int BLOCKHEIGHT =  32;

  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  half2 scale = scales[col];
  half2 zero = __hneg2(zeros[col]);
  __shared__ half2 blockvec[blockwidth2];

  for (int b = 0; b < batchsize; b++) {
    if (threadIdx.x < blockwidth2)
      blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x + b * height / 4 * 16];
    int i = width * row + col;
    int k = 0;

    float res = 0;

    __syncthreads();
    unsigned int tmp1, tmp2, tmp3, tmp4=0;

    while (k < blockwidth2) {
      tmp1 = as_unsigned(mat[i]);
      i += width;
      half2 t = {};
      FragB fragB_0 = dequant_faster(tmp1);
      FragB fragB_1 = dequant_faster(tmp1 >> 6);
      tmp4 |= (tmp1 & 0xf000f000) >> 12;
      t = __hfma2(__hfma2(fragB_0[0], scale, zero), blockvec[k + 0], t);
      t = __hfma2(__hfma2(fragB_0[1], scale, zero), blockvec[k + 1], t);    
      t = __hfma2(__hfma2(fragB_1[0], scale, zero), blockvec[k + 2], t);
      t = __hfma2(__hfma2(fragB_1[1], scale, zero), blockvec[k + 3], t);    

      tmp2 = as_unsigned(mat[i]);
      i += width;
      FragB fragB_2 = dequant_faster(tmp2);
      FragB fragB_3 = dequant_faster(tmp2 >> 6);
      tmp4 |= (tmp2 & 0xf000f000) >> 8;
      t = __hfma2(__hfma2(fragB_2[0], scale, zero), blockvec[k + 4], t);
      t = __hfma2(__hfma2(fragB_2[1], scale, zero), blockvec[k + 5], t);    
      t = __hfma2(__hfma2(fragB_3[0], scale, zero), blockvec[k + 6], t);
      t = __hfma2(__hfma2(fragB_3[1], scale, zero), blockvec[k + 7], t);
      

      tmp3 = as_unsigned(mat[i]);
      i += width;
      FragB fragB_4 = dequant_faster(tmp3);
      FragB fragB_5 = dequant_faster(tmp3 >> 6);
      tmp4 |= (tmp3 & 0xf000f000) >> 4;
      t = __hfma2(__hfma2(fragB_4[0], scale, zero), blockvec[k + 8], t);
      t = __hfma2(__hfma2(fragB_4[1], scale, zero), blockvec[k + 9], t);    
      t = __hfma2(__hfma2(fragB_5[0], scale, zero), blockvec[k + 10], t);
      t = __hfma2(__hfma2(fragB_5[1], scale, zero), blockvec[k + 11], t);


      FragB fragB_6 = dequant_faster(tmp4);
      FragB fragB_7 = dequant_faster(tmp4 >> 6);
      t = __hfma2(__hfma2(fragB_6[0], scale, zero), blockvec[k + 12], t);
      t = __hfma2(__hfma2(fragB_6[1], scale, zero), blockvec[k + 13], t);    
      t = __hfma2(__hfma2(fragB_7[0], scale, zero), blockvec[k + 14], t);
      t = __hfma2(__hfma2(fragB_7[1], scale, zero), blockvec[k + 15], t);
      res += __half2float(__hadd(t.x ,t.y));
      k += 16;
    }

    atomicAdd(&mul[col + b * width], res);
  }
}

