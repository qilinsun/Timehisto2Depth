# coding=utf-8
import timeit

import scipy.io as scio
import numpy as np
import numba
import os
import imageio
import time


# SPAD Parameters

DCR = 3000             # Dark Count Rate (counts/s)
PDP = 0.3              # Photon Detection Probability (-)
APP = 0.01             # After Pulsing Probability (-)
CTP = 0.001            # Crosstalk Probability (-)
t_dead = 10e-9         # Dead time (s)
noise_back = 5e11      # Background noise (counts/s)

c = 3e8                # Speed of light (m/s)
exp = 0.005            # Film exposition (m)
res_xy = 300           # Space Resolution (pixels)
res_t = 4096           # Time Resolution (pixels)


# 'counts_1', 'counts_2', 't_1', 't_2', 'mu_noise_1', 'mu_noise_2', 'std_noise_1', 'std_noise_2'
jitter_mat_data = scio.loadmat('jitter.mat')
mu_noise = jitter_mat_data['mu_noise_1']
counts = jitter_mat_data['counts_1']
t = jitter_mat_data['t_1']
t_1 = jitter_mat_data['t_1']
counts_1 = jitter_mat_data['counts_1']

M = int(1e4)                # Measurements

## Processing and Simulation

dt = exp/c
folder_in = 'hdr_render'     # Render streaks
folder_out = 'hdr_spad'      # SPAD streaks

if not os.path.exists(folder_out):
    os.mkdir(folder_out)

hdr_spad = np.empty([res_xy, res_xy, res_t], dtype=np.double)

imageio.plugins.freeimage.download()  #needed when running for the first time


# y = randsample(population,k,true,w) 使用与向量 population 长度相同的非负权重向量 w 来确定值 population(i) 被选为 y 的输入项的概率。
def matlab_randsample(population: np.ndarray, k: int, w: np.ndarray):
    w_sum = np.sum(w)
    normalized_w = w / w_sum
    return np.random.choice(population, k, replace=True, p=normalized_w)

rand = np.random.rand
def cal():
    for i in range(res_xy): # 300
        hdr_path = os.path.join(folder_in, f"img_{i:04}.hdr")
        hdr = imageio.imread(hdr_path, format='HDR-FI')
        hdr = hdr[:, :, 0] # Get only one RGB channel for speedup (comment if needed)
        W, H = hdr.shape
        start = time.time()
        for j in range(W): # 300
            k_vec = matlab_randsample(np.arange(0, H), M, hdr[j, :])
            jitter_vec = np.round(matlab_randsample(t.flatten(), M, counts_1.flatten()) / dt).astype(np.int32)
            idx = 1
            for k in k_vec:  # 10000
                if hdr[j, k] != 0 and rand() < PDP:  # Photon detected
                    jitter = jitter_vec[idx]
                    idx += 1
                    if k + jitter < 0:
                        hdr_spad[i, j, 0] += 1
                    elif k + jitter > H:
                        hdr_spad[i, j, -1] += 1
                    else:
                        hdr_spad[i, j, k + jitter] += 1
                    # Dead time + Afterpause
                    afterpause = round(t_dead/dt)
                    n = 1
                    while k + afterpause <= H:
                        if rand() < np.power(APP, n):
                            hdr_spad[i, j, k + afterpause] += 1
                        afterpause = afterpause + round(t_dead / dt)
                        n += 1
                    # Crosstalk left
                    if rand() < CTP and j - 1 > 0:
                        hdr_spad[i, j - 1, k] += 1
                    if rand() < CTP and j + 1 < W:
                        hdr_spad[i, j + 1, k] += 1
                    if rand() < CTP and i + 1 < W:
                        hdr_spad[i + 1, j, k] += 1
                    if rand() < CTP and i - 1 > 0:
                        hdr_spad[i - 1, j, k] += 1

            print(f"Line {i} -  {float(j) / W * 100} %")
        print(time.time() - start)
        mu_noise_back = mu_noise * M / np.sum(counts) * H / t.shape[0]
        RTN = np.random.poisson(mu_noise_back, (1, W, H))
        hdr_spad[i, :, :] = hdr_spad[i, :, :] + RTN




print(timeit.timeit(cal, number=1))
