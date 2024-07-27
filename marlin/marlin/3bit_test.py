import numpy as np
import torch
import torch.nn as nn

import marlin

DEV = torch.device('cuda:0')
K = 128
N = 256

DTYPE = torch.half
vec = torch.ones((1, K), device=DEV, dtype=DTYPE)
mul = torch.zeros((1, N), device=DEV, dtype=DTYPE)

def test_4bit():
    # mat = torch.randint(-1000000000, 1000000000, (M // 16, N*16//8), device=DEV, dtype=torch.int)
    np_array = np.full((K // 16, N*16//8), 0x99999999, dtype=np.int32)
    mat = torch.from_numpy(np_array).to('cuda' if torch.cuda.is_available() else 'cpu')
    # print(mat)
    scales = torch.ones((1,N), device=DEV, dtype=DTYPE)
    workspace = torch.zeros(N // 128 * 16, device=DEV)
    marlin.mul(vec, mat, mul, scales, workspace, 128, 128, -1)
    torch.cuda.synchronize()
    return mul

def test_3bit():
    # mat = torch.randint(-1000000000, 1000000000, (M // 16, N*16//8), device=DEV, dtype=torch.int)
    np_array = np.full((K // 16, N*16//8), 0x99999999, dtype=np.int32)
    mat = torch.from_numpy(np_array).to('cuda' if torch.cuda.is_available() else 'cpu')
    # print(mat)
    scales = torch.ones((1,N), device=DEV, dtype=DTYPE)
    workspace = torch.zeros(N // 128 * 16, device=DEV)
    marlin.mul_3bit(vec, mat, mul, scales, workspace, 128, 128, -1)
    torch.cuda.synchronize()
    return mul

def gen_quant3(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 3 - 1
    w = torch.ones((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    #w = torch.clamp(w, 0, maxq)
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
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
    layer = marlin.Layer3bit(m,n, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B1 = torch.empty((m // 16, n * 16 * 2 // 32), dtype=torch.int, device=DEV)
    layer.B2 = torch.empty((m // 16, n * 16  // 32), dtype=torch.int, device=DEV)
    #print('% 5d % 6d' % (m // 16, n * 16 * 3 // 32))
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q1 = layer.B1
    q2 = layer.B2
    s = layer.s
    return ref, q1, q2, s

def run_problem(m, n, k, thread_k, thread_n, groupsize=-1):
    print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
    A = torch.ones((m, k), dtype=torch.half, device=DEV)
    B_ref, B1, B2, s = gen_quant3(k, n, groupsize=groupsize)
    C = torch.zeros((m, n), dtype=torch.half, device=DEV)
    C_ref = torch.matmul(A, B_ref)
    workspace = torch.zeros(n // 128 * 16, device=DEV)
    marlin.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, -1)
    torch.cuda.synchronize()
    print(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)))

run_problem(16, 2 * 256, 768, 64, 256)

'''
mul = test_3bit()
print(mul)
mul = test_4bit()
print(mul)
'''