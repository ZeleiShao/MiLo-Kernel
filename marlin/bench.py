import sys

import numpy as np
import torch
import torch.nn as nn
import marlin  # Make sure to import marlin first

# Get the location of the marlin module
module_location = sys.modules['marlin'].__file__
print(module_location)

import time

def benchmark(f, warmup=20, iter=100):
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

def gen_quant3(m, n, groupsize=-1, tile_shape=0):
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
    z = torch.randn(s.shape)
    #ref = w.half() * s + z
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
    #layer = marlin.Layer3bit(m, n, groupsize=groupsize, tile_shape = tile_shape)
    #layer = marlin.Layer3bitFaster(m, n, groupsize=groupsize)
    #layer = marlin.Layer3bitWithZero(m, n, groupsize=groupsize)
    #layer = marlin.Layer3bit256_64(m, n, groupsize=groupsize)
    #layer = marlin.Layer3bit256_64_with_zero(m, n, groupsize=groupsize)
    layer = marlin.Layer3bit_64_256_WithZero(m, n, groupsize=groupsize)

    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B1 = torch.empty((m // 16, n * 16 * 2 // 32), dtype=torch.int, device=dev)
    layer.B2 = torch.empty((m // 16, n * 16 // 32), dtype=torch.int, device=dev)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=dev)
    layer.z = torch.empty((m // groupsize, n), dtype=torch.half, device=dev)
    #layer.pack(linear, s.t(), z.t())
    layer.pack(linear, s.t(), z.t())
    #layer.pack(linear, s.t())
    #layer.pack(linear, s.t())
    q1 = layer.B1
    q2 = layer.B2
    s = layer.s
    z = layer.z
    return ref, q1, q2, s, z


def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B1, B2, C, s, z,thread_k, thread_n, sms):
    workspace = torch.zeros((C.shape[1] // 128) * 16, device=torch.device('cuda:0'))
    #res = benchmark(lambda: marlin.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, sms))
    #res = benchmark(lambda: marlin.mul_3bit_faster(A, B1, B2, C, s, workspace, thread_k, thread_n, sms))
    #res = benchmark(lambda: marlin.mul_3bit_256_64(A, B1, B2, C, s, workspace, 256, 64, sms))
    #res = benchmark(lambda: marlin.mul_3bit_256_64_with_zero(A, B1, B2, C, s,z, workspace, 256, 64, sms))
    #res = benchmark(lambda: marlin.mul_3bit_with_zero(A, B1, B2, C, s, z, workspace, thread_k, thread_n, sms))
    res =  benchmark(lambda: marlin.mul_3bit_64_256_with_zero(A, B1, B2, C, s, z, workspace,64, 256, sms))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel() +  2 * z.numel()) / res / 10 ** 9

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
#    'ideal': [
#        (4 * 256 * SMS, 256 * SMS)
# ],
    'deepseek' : [(2048, 11008),(11008,2048),(11008,2048)],
   'mixtual' : [(4096, 14336),(14336, 4096),(4096, 14336)], 
   'arctic' : [(7168,4864),(4864, 7168),(4864, 7168)],
   'Falcon180B1' : [(14848, 14848 * 5 + 1024),(14848 * 5, 14848)],

   'deepseek1' : [(2048, 11008)],
   'deepseek2' : [(11008,2048)],
   'mixtual1' : [(4096, 14336)], 
   'mixtual2' : [(14336, 4096)],
   'arctic1' : [(4864, 7168)],
   'artic2' : [(7168,4864)],
    'Falcon180B1' : [(14848, 14848 * 5 + 1024)],
    'falocon190B2' : [(14848 * 5, 14848)]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments


#2.43, 3.44
for groupsize in [64] :
    print('groupsize=%d' % groupsize)
    print()
    dev = torch.device('cuda:0')
    for model, layers in MODELS.items():
        print(model)
        
        batchsizes = [1, 16, 32]
        for batch in batchsizes: 
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory(MB)' : 0}  
            tot_d = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory(MB)' : 0}  
            
            for layer in layers:
                #B_ref, B1, B2, s, z = gen_quant3(layer[0], layer[1], groupsize, tile_shape)
                tile_shape = 0
                if layer[0] > layer[1] * 2 and layer[0] < 17000 and batch <= 16:
                    tile_shape = 1
                A, B1, B2, C, B_ref, s, z = get_problem_int3(batch, layer[1], layer[0], groupsize, tile_shape)
                #A, B, C, B_ref, s =  get_problem(batch, layer[1], layer[0], groupsize=128)
                res_d = benchmark_dense(A, B_ref, C)
                #if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                #res_q = benchmark_quant_4bit(A, B, C, s, 64, 256, SMS)
                #res_q = benchmark_quant(A, B1, B2, C, s,z, -1, -1, SMS)
                #else:
                #res_q = benchmark_quant_4bit(A, B, C, s, -1, -1, SMS)
                res_q = benchmark_quant(A, B1, B2, C, s,z, 64, 256, SMS)

                #res_q = benchmark_quant4bit(A, B, C, s, z, -1, -1, SMS)

                res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                tot_q['memory(MB)'] += (4 * B1.numel() + 4 * B2.numel() + 2 * s.numel()) / 10 ** 6
                for k in tot_q:
                    if k != 's' and k != 'memory(MB)' :
                        tot_q[k] += res_q[k] * res_q['s']
                  
            for k in tot_q:
                if k != 's' and k != 'memory(MB)':
                    tot_q[k] /= tot_q['s']
            print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f, memory(MB)=%.2f, parametersnum : %.2f' % (
                batch,
                tot_q['s'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup'],
                tot_q['memory(MB)'],
                layers[0][0] * layers[0][1] / 10**9
            ))
        print()
