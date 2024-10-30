import torch
import torch.nn as nn

import deGE_cuda
print(deGE_cuda.__file__)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_speed(M,N,K):
    COUNT = 300
    import time
    DTYPE = torch.half
    DEV = torch.device('cuda:0')
    mat = torch.randn((K,N), device=DEV, dtype=DTYPE)
    vec = torch.randn((M,K), device=DEV, dtype=DTYPE)
    mul = torch.zeros((M, N), device=DEV, dtype=DTYPE)

    tick = time.time()
    for _ in range(COUNT):
        torch.matmul(vec, mat, out=mul) 
        torch.cuda.synchronize()
    time_fp16 = (time.time() - tick) / COUNT

    DTYPE = torch.float
    mat = mat.to(DTYPE)
    vec = vec.to(DTYPE)
    mul = mul.to(DTYPE)

    mat = torch.randint(-1000000000, 1000000000, (K // 1024 * 96, N), device=DEV, dtype=torch.int)
    scales = torch.randn(N, device=DEV, dtype=DTYPE) #per channel quantization
    zeros = torch.randn(N, device=DEV, dtype=DTYPE)

    COUNT = 1000
    import time
    tick = time.time()
    for _ in range(COUNT):
        deGE_cuda.Batchvecquant3matmul_faster(vec, mat, mul, scales, zeros)
        torch.cuda.synchronize()
    time_3bit = (time.time() - tick) / COUNT
    return time_fp16 , time_3bit, 2 * K * M * N / 10**12

MODELS = {
   'deepseek' : [(2048, 11008),(11008,2048),(11008,2048)],
   'mixtual' : [(4096, 14336),(14336, 4096),(4096, 14336)], 
   'arctic' : [(7168,4864),(4864, 7168),(4864, 7168)],
    'Falcon180B1' : [(14848, 14848 * 5 + 1024),(14848 * 5, 14848)]
}

for model, layers in MODELS.items():
    print(model)
    for batchsize in [1, 16, 32] : 
        tot_fp16 = 0.0
        tot_3bit = 0.0
        tot_cal = 0.0
        fp16 = 0.0
        w3 = 0.0
        for layer in layers:
            fp16, w3, cal = get_speed(batchsize, layer[1],layer[0])
            tot_fp16 += fp16 
            tot_3bit += w3 
            tot_cal += cal

        TFLOPS = tot_cal / tot_3bit
        print("batchsize=%d, TFLOP/s=%.2f "%(batchsize,TFLOPS))

