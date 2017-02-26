#!/usr/bin/env python
#All kinds of test functions to debug, prob the cnn
import numpy as np
import math

def flatten_weights(mat):
    N = mat.shape[0]
    C = mat.shape[1]
    H = mat.shape[2]
    W = mat.shape[3]
    flat_W_float = math.sqrt(N * C)
    flat_W = 0
    for w in range(1, int(flat_W_float)):
        if ((N * C) % w) == 0:
            flat_W = w
    flat_H = N * C/ flat_W;
    w_flat = np.zeros((flat_H * H + flat_H - 1, flat_W * W + flat_W - 1))
    w_flat[:] = np.nan
    for i in range(N * C):
        h = i / flat_W
        w = i % flat_W
        w_flat[(H + 1) * h : (H + 1) * (h + 1) - 1, (W + 1) * w : (W + 1) * (w + 1) - 1]=mat[i / C, i % C, :, :]
    return w_flat
