const int4* __restrict__ B1, // 3bit quantized weight matrix of shape kxn
const int4* __restrict__ B2,



 extern __shared__ int4 sh[];
 // Shared memory storage for global fetch pipelines.
 int4* sh_a = sh;
 int4* sh_b1 = sh_a + stages * a_sh_stage;
 int4* sh_b2 = sh_b1 + stages*b_sh_stage/2;

 I2 frag_b1_quant[2];
 int frag_b2_quant[2];

const int4* B1_ptr[b_sh_wr_iters];
const int4* B2_ptr[b_sh_wr_iters];

#pragma unroll
for (int i = 0; i < b_sh_wr_iters/2; i++)
  {
    B1_ptr[i] = B1 + b_gl_rd_delta_i * i + b_gl_rd;
    //B2_ptr[i] = B2 + b_gl_rd_delta_i * i + b_gl_rd;
  }

auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
        int4* sh_b1_stage = sh_b1 + b_sh_stage/2 * pipe;
        int* sh_b2_stage = sh_b2 + b_sh_stage * pipe;
      if (2 * threadIdx.x < threads){
      #pragma unroll
      for (int i = 0; i < b_sh_wr_iters/2; i++) {
       cp_async4_stream(&sh_b1_stage[b_sh_wr_delta * i + b_sh_wr], B1_ptr[i]);
       //cp_async_stream1(&sh_b2_stage[b_sh_wr_delta * i + b_sh_wr], B2_ptr[i]);
       B1_ptr[i] += b_gl_rd_delta_o;
       //B2_ptr[i] += b_gl_rd_delta_o;
       }
     }   

    
   // Insert a fence even when we are winding down the pipeline to ensure that waiting is also correct at this point.
   cp_async_fence();
 };
}




