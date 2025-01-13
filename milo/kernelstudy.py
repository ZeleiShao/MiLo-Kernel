import sys
import numpy as np
import torch
import torch.nn as nn
import milo_kernel  # Make sure to import marlin first


import time

def benchmark(f, warmup=20, iter=500):
    for i in range(warmup + iter):
        tick = time.time()
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = 1000000 * (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(5.)
    return res

def get_problem(m, n, k, groupsize=128):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.ones((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def get_problem_int3(m, n, k, groupsize=64, tile_shape=0):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B1 = torch.randint(low=-2**31, high=2**31, size=(k * n // 16,), device=dev)
    B2 = torch.randint(low=-2**31, high=2**31, size=(k * n // 32,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    z = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B1, B2, C, B_ref, s, z

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        'mus': res,
        'TFLOP': 2 * A.numel() * C.shape[1]/ 10 ** 12,
        'GB': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / 10 ** 9
    }

def benchmark_quant(A, B1, B2, C, s, z,thread_k, thread_n, sms):
    workspace = torch.zeros((C.shape[1] // 128) * 16, device=torch.device('cuda:0'))
    #res = benchmark(lambda: milo_kernel.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, sms))
    res = benchmark(lambda: milo_kernel.mul_3bit_faster(A, B1, B2, C, s, workspace, 64, 256, sms))
    #res = benchmark(lambda: milo_kernel.mul_3bit_256_64(A, B1, B2, C, s, workspace, 256, 64, sms))
    #res = benchmark(lambda: milo_kernel.mul_3bit_256_64_with_zero(A, B1, B2, C, s,z, workspace, 256, 64, sms))
    #res = benchmark(lambda: milo_kernel.mul_3bit_with_zero(A, B1, B2, C, s, z, workspace, thread_k, thread_n, sms))
    #res = benchmark(lambda: milo_kernel.mul_3bit_64_256_with_zero(A, B1, B2, C, s, z, workspace,64, 256, sms))
    return {
        'mus': res,
        'TFLOP': 2 * A.numel() * C.shape[1] / 10 ** 12,
        'GB': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel() +  2 * z.numel()) / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1

MODELS = { #(k,n)
#    'ideal': [
#        (4 * 256 * SMS, 256 * SMS)
# ],
#    'deepseek' : [(2048, 11008),(11008,2048),(2048, 11008)],
#   'mixtral' : [(4096, 14336),(14336, 4096),(4096, 14336)], 
#     'arctic' : [(7168,4864),(4864, 7168),(7168,4864)],
#    'Llama65B': [
#         (8192, 3 * 8192),
#         (8192, 8192),
#         (8192, 2 * 21760),
#         (21760, 8192)
#     ],
#   'Falcon180B' : [(14848, 14848 * 5 + 1024),(14848 * 5, 14848)],

# 'deepseek1' : [(2048, 11008)],
# 'deepseek2' : [(11008,2048)],
# 'arctic1' : [(4864, 7168)],
# 'artic2' : [(7168,4864)],
# 'mixtral1' : [(4096, 14336)], 
# 'mixtral2' : [(14336, 4096)],
'Falcon180B1' : [(14848, 14848 * 5 + 1024)],
'Falocon180B2' : [(14848 * 5, 14848)]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments


#2.43, 3.44
for groupsize in [64] :
    print()
    dev = torch.device('cuda:0')
    for model, layers in MODELS.items():
        print(model)
        
        batchsizes = [32]
        for batch in batchsizes: 
            for sms in [108]: 
                tot_q = {'mus': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory' : 0, 'TFLOP': 0}  
                tot_d = {'mus': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory' : 0, 'TFLOP': 0}  

                tile_shape = 0
                A, B1, B2, C, B_ref, s, z = get_problem_int3(batch, layers[0][1], layers[0][0], groupsize, tile_shape)
                workspace = torch.zeros((C.shape[1] // 128) * 16, device=torch.device('cuda:0'))
                #milo_kernel.mul_3bit_faster(A, B1, B2, C, s, workspace, 64, 256, sms)
                res_q = benchmark_quant(A, B1, B2, C, s, z, 64, 256, sms)
                tot_q['mus'] += res_q['mus']
                tot_q['memory'] += res_q['GB']
                tot_q['TFLOP'] += res_q['TFLOP']
            print('batch=%04d: mus=%.5f, speedup=%.2f, fp16=%.5f' % (
                batch,
                tot_q['mus'],
                tot_q['speedup'],
                tot_d['mus']
                ))
        print()

