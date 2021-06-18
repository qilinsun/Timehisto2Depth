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
#### main_v1_8400
基本上1:1用python翻译了matlab，用于检查正确性，非常慢，顾名思义大概要跑8400秒
#### main_v2_numba_cpu_220
所有计算函数使用numba加速，运行时编译成机器码，大概需要220秒。
使用argmax寻找峰值的函数试图重建了一下，看起好像没啥问题，需要进一步检查
#### main_v3_numba_cpu_165
预处理APP这个参数的N次方，降低到165秒
### 多核心纯CPU
### 多核心CPU和GPU混合

## 没啥用的小工具
## cal_md5.py
快速计算上百G文件的MD5
