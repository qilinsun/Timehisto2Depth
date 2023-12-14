# coding=utf-8
import threading
import scipy.io as scio
import numpy as np
import numba
import os
import time
import h5py
import queue
import imageio
from multiprocessing.pool import ThreadPool


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

# Processing and Simulation
dt = exp/c
folder_in = 'hdr_render'     # Render streaks
folder_out = 'hdr_spad'      # SPAD streaks

if not os.path.exists(folder_out):
    os.mkdir(folder_out)

imageio.plugins.freeimage.download()  # needed when running for the first time

lock = threading.Lock()

def read_hdr(i : int, file_path : str, hdrs : np.ndarray):
    hdr = imageio.imread(file_path, format='HDR-FI')
    lock.acquire()
    hdrs[i, ...] = hdr
    lock.release()


def read_hdrs():
    hdrs = np.empty((res_xy, res_xy, res_t))  # (300, 300, 4096, 3)
    hdr_paths = [os.path.join(folder_in, f"img_{i:04}.hdr") for i in range(res_xy)]
    pool = ThreadPool(8)
    params = [(i, hdr_paths[i], hdrs) for i in range(res_xy)]
    pool.starmap(read_hdr, params)
    pool.close()
    pool.join()
    return hdrs


# y = randsample(population,k,true,w) 使用与向量 population 长度相同的非负权重向量 w 来确定值 population(i) 被选为 y 的输入项的概率。
def matlab_randsample(population: np.ndarray, k: int, w: np.ndarray):
    w_sum = np.sum(w)
    normalized_w = w / w_sum
    return np.random.choice(population, k, replace=True, p=normalized_w)


rand = np.random.rand
hdrs = read_hdrs()


@numba.njit(cache=True)
def rand_choice_nb(population, k, w):
    """
    :param population: A 1D numpy array of values to sample from.
    :param k: number of samples to chose from population
    :param w: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    w_sum = np.sum(w)
    normalized_w = w / w_sum
    cum_normalized_w = np.cumsum(normalized_w)
    cum_normalized_w[-1] = 1.0
    rands = np.random.random(k)
    res = np.empty(k, np.float64)
    for i in range(k):
        # returned index i satisfies a[i-1] < v <= a[i]
        res[i] = population[np.searchsorted(cum_normalized_w, rands[i], side="left")]
    return res


@numba.njit((numba.float64, numba.int64, numba.int64), cache=True)
def nb_poisson(mu, W, H):
    noise = np.empty((W, H), np.float64)
    for i in range(W):
        for j in range(H):
            noise[i, j] = np.random.poisson(mu)
    return noise


@numba.njit(cache=True)
def cal(hdrs):
    hdr_spad = np.empty((res_xy, res_xy, res_t), np.float64)
    for i in range(res_xy):  # 300
        hdr = hdrs[i, :, :, 0] # Get only one RGB channel for speedup (comment if needed)
        W, H = hdr.shape
        for j in range(W):  # 300
            k_vec = rand_choice_nb(np.arange(0, H), M, hdr[j, :]).astype(np.int64)
            jitter_vec = rand_choice_nb(t.flatten(), M, counts_1.flatten()) / dt
            rounded_jitter_vec = np.empty_like(jitter_vec)
            np.round_(jitter_vec, 0, rounded_jitter_vec)
            rounded_jitter_vec = rounded_jitter_vec.astype(np.int64)
            idx = 1
            for k in k_vec:  # 10000
                if hdr[j, k] != 0 and rand() < PDP:  # Photon detected
                    jitter = rounded_jitter_vec[idx]
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
                    # Crosstalk right
                    if rand() < CTP and j + 1 < W:
                        hdr_spad[i, j + 1, k] += 1
                    # Crosstalk top
                    if rand() < CTP and i + 1 < W:
                        hdr_spad[i + 1, j, k] += 1
                    # Crosstalk bottom
                    if rand() < CTP and i - 1 > 0:
                        hdr_spad[i - 1, j, k] += 1

            #print("Line", str(i), " - ", j * 100 / W, "%")
        print("Line", str(i))
        mu_noise_back = (mu_noise * M / np.sum(counts) * H / t.shape[0])[0][0]
        RTN = nb_poisson(mu_noise_back, W, H)
        hdr_spad[i, :, :] = hdr_spad[i, :, :] + RTN
    return hdr_spad

def save_h5(hdr_spad, filename):
    print("save hdr_spad to ", filename)
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("hdr_spad", hdr_spad.shape, dtype='f8', compression="gzip", compression_opts=4)
        dset[()] = hdr_spad


@numba.njit(cache=True)
def get_depth_map(hdr_spad):
    W, H, T = hdr_spad.shape
    depth_map = np.empty((W, H), np.float64)
    for i in range(W):
        for j in range(H):
            depth_map[i, j] = np.argmax(hdr_spad[i, j, :])
    depth_map = depth_map - np.min(depth_map)
    depth_map = depth_map / np.max(depth_map)
    depth_map = 1 - depth_map
    return depth_map


def write_depth_map(depth_map, filename):
    print("write the depth map to ", filename)
    imageio.imwrite(filename, depth_map)


if __name__ == "__main__":
    start = time.time()
    hdr_spad = cal(hdrs)
    #scio.savemat('depth_map.mat',{'depth_map' : depth_map})
    print(time.time() - start)

    save_h5(hdr_spad, "hdr_spad.h5")
    depth_map = get_depth_map(hdr_spad)
    write_depth_map(depth_map, "depth_map_argmax.png")
