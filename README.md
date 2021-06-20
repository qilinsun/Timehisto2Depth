# TODOs
[] Read 512*512 SPAD paper, extract useful parameters for simulation.
[] Be familar with the mode of SPAD camera, write a reading notes of your understandings.
[] Write a plan about this project

# Timehisto2Depth
## 下载必要的资源文件
链接：https://pan.baidu.com/s/1GZof-keTT-QXbv1w8vLv4g 
提取码：snva   
下载完成后放在工程, 根目录
## 环境
我这边使用的python3.9目测3.7+应该没有问题  
环境使用anaconda3配置，有空了可以打包个docker镜像  
### conda的国内镜像
[推荐使用北京外国语大学的TUNA镜像](https://mirrors.bfsu.edu.cn/help/anaconda/)
### numba
运行必须的库，要不然性能无法接受
```
conda install numba
```
### imageio
用于读取和写入图片
```
conda install -c conda-forge cupy cudatoolkit=10.1 cudnn cutensor nccl
```
### h5py
用于读取和写入h5文件
```
conda install h5py
```
### scipy
用于读取和写入mat文件
```
conda install scipy
```
### cupy
GPU加速库。安装的时候注意cudatoolkit版本不一定是10.1，务必要和电脑上装的cuda runtime版本一致。nccl没有也不要紧，要是卡特别多会稍有点影响。
```
conda install -c conda-forge cupy cudatoolkit=10.1 cudnn cutensor nccl
```
## 运行环境
## CPU
E5 2678V3 12核心24线程，2.5G 很旧
## 内存
32G双通道DDR3 REGECC
## 如果出现GPU
有时候是V100，有时候是V100 MAX-Q，大概相差20%性能，显存都为32G
## 硬盘
铠侠RD20 一个性能很不错的固态，开24个线程的话大概有1500MB/S读写

## 各版本详情
### 单核心纯CPU
#### 1. main_v1_8400.py
基本上1:1用python翻译了matlab，用于检查正确性，非常慢，顾名思义大概要跑8400秒
#### 2. main_v2_numba_cpu_220.py
所有计算函数使用numba加速，运行时编译成机器码，大概需要220秒。
使用argmax寻找峰值的函数试图重建了一下，看起好像没啥问题，需要进一步检查
#### 3. main_v3_numba_cpu_165.py
预处理APP这个参数的N次方，降低到165秒
#### 4. main_v3.5_numba_cpu_144.py
```
k_vec = randsample(1:size(hdr,2),M,true,hdr(j,:,1));    % Importance sampling
jitter_vec = round(randsample(t_1,M,true,counts_1)/dt); idx = 1;
```
观察一下两个重要性采样，发现`1:size(hdr,2)`、`round(t_1/dt)`、`round(counts_1/dt)`都是定值，可以预先计算。所有矩阵能预先分配空间的就预先分配空间，避免动态内存分配。运算时间降低到144秒。

### 多核心纯CPU
#### 1. main_v4_numba_multi_cpus_89.py
随机采样10000个样本可以并行。并行这个函数，运算时间降低到89秒。注意MAX_THREAD最大应该设置为物理核心数。
### 多核心CPU和GPU混合
#### 1. GPU处理权重和随机数，CPU使用预处理的权重和随机数来计算
GPU预处理所有按权重采样需要的权重，并生成35亿个double随机数，然后从显存传回内存，共需要7秒。
CPU开8个进程，共需要58秒。因为GPU和CPU可以同时计算，所以实际上只需要58秒。


## 没啥用的小工具
## cal_md5.py
快速计算上百G文件的MD5
