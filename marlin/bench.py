import sys

import numpy as np
import torch
import torch.nn as nn
import marlin

import time

def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res

def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B1 = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B2 = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s


def gen_quant3(m, n, groupsize=-1):
    maxq = 2 ** 3 - 1
    dev = torch.device('cuda:0')
    w = torch.randn((m, n), dtype=torch.half, device=dev)

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    #ref = w
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    #layer = marlin.Layer3bit(256, 256, groupsize=groupsize)
    layer = marlin.Layer3bit(m, n, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B1 = torch.empty((m // 16, n * 16 * 2 // 32), dtype=torch.int, device=dev)
    layer.B2 = torch.empty((m // 16, n * 16 // 32), dtype=torch.int, device=dev)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=dev)
    layer.pack(linear, s.t())
    q1 = layer.B1
    q2 = layer.B2
    s = layer.s
    return ref, q1, q2, s




def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B1, B2, C, s, thread_k, thread_n, sms):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: marlin.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, sms))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
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

MODELS = {
    #'ideal': [
    #    (4 * 256 * SMS, 256 * SMS)
    #],
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

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
ALL = False

#for groupsize in [-1, 128] if ALL else [128]:
for groupsize in [-1, 128] if ALL else [128]:
    print('groupsize=%d' % groupsize)
    print()
    dev = torch.device('cuda:0')
    for model, layers in MODELS.items():
        print(model)
        if ALL:
            batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        else:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for batch in batchsizes:
            if not ALL and model != 'ideal' and batch != 16:
                continue
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0} 
            for layer in layers:
                A = torch.randn((batch, layer[0]), dtype=torch.half, device=dev)
                B_ref, B1, B2, s = gen_quant3(layer[0], layer[1], groupsize)
                C = torch.zeros((batch,layer[1]), dtype=torch.half, device=dev)
                res_d = benchmark_dense(A, B_ref, C)
                if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                    res_q = benchmark_quant(A, B1, B2, C, s, 64, 256, SMS)
                else:
                    res_q = benchmark_quant(A, B1, B2, C, s, -1, -1, SMS)
                res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] += res_q[k] * res_q['s']
            for k in tot_q:
                if k != 's':
                    tot_q[k] /= tot_q['s']
            print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                batch,
                tot_q['s'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup']
            ))
        print()
