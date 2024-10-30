import torch
import torch.nn as nn

import quant_cuda
print(quant_cuda.__file__)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_speed(M,N):
    COUNT = 1000
    import time
    DTYPE = torch.half
    DEV = torch.device('cuda:0')
    mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
    vec = torch.randn((1, M), device=DEV, dtype=DTYPE)
    mul = torch.zeros((1, N), device=DEV, dtype=DTYPE)

    tick = time.time()
    for _ in range(COUNT):
        torch.matmul(vec, mat, out=mul) 
        torch.cuda.synchronize()
    time_fp16 = (time.time() - tick) / COUNT

    DTYPE = torch.float
    mat = mat.to(DTYPE)
    vec = vec.to(DTYPE)
    mul = mul.to(DTYPE)

    mat = torch.randint(-1000000000, 1000000000, (M // 1024 * 96, N), device=DEV, dtype=torch.int)
    scales = torch.randn(N, device=DEV, dtype=DTYPE) #per channel quantization
    zeros = torch.randn(N, device=DEV, dtype=DTYPE)

    COUNT = 1000
    import time
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant3matmul_faster(vec, mat, mul, scales, zeros)
        torch.cuda.synchronize()
    time_3bit = (time.time() - tick) / COUNT
    return time_fp16 , time_3bit, 2 * M * N / 10**12

MODELS = {
   'deepseek' : [(2048, 11008),(11008,2048),(11008,2048)],
   'mixtual' : [(4096, 14336),(14336, 4096),(4096, 14336)], 
   'arctic' : [(7168,4864),(4864, 7168),(4864, 7168)],
    'Falcon180B1' : [(14848, 14848 * 5 + 1024),(14848 * 5, 14848)]
}

for model, layers in MODELS.items():
    print(model)
    tot_fp16 = 0
    tot_3bit = 0
    tot_cal = 0
    fp16 = 0
    w3 = 0
    for layer in layers:
        fp16, w3, cal = get_speed(layer[0],layer[1])
        tot_fp16 += fp16 
        tot_3bit += w3 
        tot_cal += cal

    TFLOPS = tot_cal / tot_3bit

    print("speedup=%.2f "%(TFLOPS))

