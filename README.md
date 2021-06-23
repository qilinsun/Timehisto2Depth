# TODOs
- [x] Read 512*512 SPAD paper.
- [x] Extract useful parameters for simulation.
- [x] Be familar with the mode of SPAD camera, write a reading notes of your understandings.
- [x] Write a plan about this project

# PLAN
这个有点难度

1. 搞清楚原理
   - 问孙老师一些论文里提到的不懂的概念（6.21-6.23）
   - 提取模拟的关键参数（6.21-6.23）
   - 简单写一下SPAD相机工作模式（6.21-6.23）
2. 生成模拟数据
3. 训练深度重建模型
# SPAD相机的工作模式
1. 全局快门
打开像素，关闭像素，开始读出，结束读出，串行。
每个1比特帧的总曝光量为950 ns，这是通过在10.2 µs的1比特读出之前，以400 ns的周期打开一个95 ns宽的门10次来实现的。
4400fps 4bit
274 8bit
2. 滚动快门
（打开像素，关闭像素）（开始读出，结束读出），并行，开了像素就开始逐行读。
10×95 ns宽的门。
全局曝光，1bit，97700fps


# 关键参数
## 极限参数
### 门控机制
- 打开skew = 250ps
- 关闭skew = 344ps
- 最小持续时间 = 5.75ps

### 最高帧率
97700fps 1bit
24fps 12bit

### 门偏移(gate shifts)
<40ps，这个还是不太懂，大概是每行内不同像素的门控信号并不是同时到达的

### 光子检出率(PDP)
过量偏压(V<sub>ex</sub>)越高，光子检出率越高。对波长520nm的光PDP最高。

### 暗计数率
我对暗计数率的理解是没有光子但是检出光子的概率，过量偏压(V<sub>ex</sub>)越低，暗计数率越低

### 整帧读出时间
全局曝光+滚动快门，感觉是全局打开然后逐行读出？大概是10.2µs

### 每个读出周期的最大光子数
1bit，多于1个光子不会被记录

### 不敏感？
首先，由于其全局快门模式操作，传感器在读出期间对光子不敏感。其次，在PC内部的8位数据从RAM传输到存储设备（SSD）期间，传感器的数据采集必须暂停。这里不太懂

### 热像素
25℃无传感器冷却的情况下大概1%，不过也不总是1%，有一个分布，见图9

### 信噪比？
这里不太明白

### 门窗口特性？
门窗口的特性及其在整个阵列中的均匀性是时间分辨成像的关键指标，最终限制了荧光体寿命的准确性和精确度。图12和图13展示了SwissSPAD2在两种不同设置下的门特性。在图12中，展示了472×256像素阵列上的5.75 ns门特性。这个实验是用一个790纳米的脉冲激光器进行的，脉冲重复频率（PRF）为20MHz。激光脉冲与20MHz的触发信号同步，该信号由摄像系统的FPGA生成。时间扫描是通过移动门和充电信号与FPGA的激光触发器的相位来实现的。可实现的最短门窗口为5.75ns，上升沿偏差为250ps，下降沿偏差为344ps。在图13中，介绍了472×256阵列上具有最低偏移的门的特性。



# 器件原理阅读笔记

https://www.yuque.com/docs/share/3f027146-171d-4f7a-9715-379ccb70e64d?# 精读了一下，做了个简单的翻译和整理

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
