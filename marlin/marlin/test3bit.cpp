const I2* __restrict__ B1, // 3bit quantized weight matrix of shape kxn
const int* __restrict__ B2,


 extern __shared__ int4 sh[];
 // Shared memory storage for global fetch pipelines.
 int4* sh_a = sh;
 I2* sh_b1 = reinterpret_cast<I2*>(sh_a + stages * a_sh_stage);
 int* sh_b2 = reinterpret_cast<int*>(sh_b1 + stages *b_sh_stage);


 I2 frag_b1_quant[2];
 int frag_b2_quant[2];


 int b_gl_stride = 16 * prob_n / 32; //global地读取b的两行数据之间的步长
 constexpr int b_sh_stride = 32 * thread_n_blocks / 4;//
 int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
 int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
 constexpr int b_sh_wr_delta = threads;
 constexpr int b_sh_rd_delta = threads;
 constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
 constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;
 int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
 b_gl_rd += b_sh_stride * slice_col;
 b_gl_rd += b_gl_rd_delta_o * slice_row;
 int b_sh_wr = threadIdx.x;
 int b_sh_rd = threadIdx.x;

 // Since B-accesses have non-constant stride they have to be computed at runtime; we break dependicies between subsequent accesses with a tile by maintining multiple pointers (we have enough registers), a tiny optimization.
 const I2* B1_ptr[b_sh_wr_iters];
 const int* B2_ptr[b_sh_wr_iters];
 #pragma unroll
 for (int i = 0; i < b_sh_wr_iters; i++)
 {
   B1_ptr[i] = B1 + b_gl_rd_delta_i * i + b_gl_rd;
   B2_ptr[i] = B2 + b_gl_rd_delta_i * i + b_gl_rd;
 }


auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
   if (pred) {
      I2* sh_b1_stage = sh_b1 + b_sh_stage * pipe;
     int* sh_b2_stage = sh_b2 + b_sh_stage * pipe;
     #pragma unroll
     for (int i = 0; i < b_sh_wr_iters; i++) {
       cp_async_stream2(&sh_b1_stage[b_sh_wr_delta * i + b_sh_wr], B1_ptr[i]);
       cp_async_stream1(&sh_b2_stage[b_sh_wr_delta * i + b_sh_wr], B2_ptr[i]);
       B1_ptr[i] += b_gl_rd_delta_o;
       B2_ptr[i] += b_gl_rd_delta_o;
       }
   // Insert a fence even when we are winding down the pipeline to ensure that waiting is also correct at this point.
   cp_async_fence();
 };
}

 // Load the next sub-tile from the current location in the shared memory pipe into the current register buffer.
 auto fetch_to_registers = [&] (int k, int pipe) {
   I2* sh_b1_stage = sh_b1 + b_sh_stage * pipe;
   int* sh_b2_stage = sh_b2 + b_sh_stage * pipe;
   frag_b1_quant[k % 2] = sh_b1_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd];
   frag_b2_quant[k % 2] = sh_b2_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd];
 };







