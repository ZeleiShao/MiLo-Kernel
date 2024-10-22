import torch
import torch.nn as nn

import quant_cuda

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking OPT-175B FC2 matvec ...')


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
    return time_fp16 , time_3bit

MODELS = {
    'ideal': [
        (4 * 256 * 108, 256 * 108)
 ],
#    'w1' : [(4096, 14336)],
#    'w2' : [(14336, 4096)],
 'mixtual8x7B' : [(4096, 14336),(14336, 4096),(4096, 14336)], 
 
 'Llama7B': [
        (4096, 3 * 4096),
        (4096, 4096),
        (4096, 2 * 10752),
        (10752, 4096)
    ],
    'Llama13B': [
        (5120, 3 * 5120),
        (5120, 5120),
        (5120, 2 * 13568),
        (13568, 5120)
    ],
    'Llama33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    'Llama33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    'Llama65B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ],
    'Falcon180B': [
        # Note that parallel attention and FC allows layer fusions
        (14848, 14848 * 5 + 1024),
        (14848 * 5, 14848)
    ]
}

for model, layers in MODELS.items():
    print(model)
    tot_fp16 = 0
    tot_3bit = 0
    fp16 = 0
    w3 = 0
    for layer in layers:
        fp16, w3 = get_speed(layer[0],layer[1])
        tot_fp16 += fp16 
        tot_3bit += w3 

    speedup = tot_fp16 / tot_3bit

    print("speedup=%.2f "%(speedup))

