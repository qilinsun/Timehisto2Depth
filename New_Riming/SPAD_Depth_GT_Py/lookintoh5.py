import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re


# os.path.append("/media/cuhksz-aci-03/数据/2rendered3D/deeptofrefdepth/DeepToF_ref_depth")
# from PIL import Image
directory = '/media/cuhksz-aci-03/数据/2rendered3D/deeptofrefdepth/DeepToF_ref_depth'
# 使用glob模块匹配所有.h5文件
h5_files = glob.glob(os.path.join(directory, '*.h5'))

# 遍历每一个.h5文件
for h5_file in h5_files:
    print(f'正在处理文件: {h5_file}')
    # 打开HDF5文件
    with h5py.File(h5_file, 'r') as f:
        # 获取数据集的名字列表
        datasets = list(f.keys())

        # 遍历数据集
        for dataset in datasets:
            print(f"Dataset: {dataset}")

            # 读取数据集
            data = f[dataset][()]

            # 将数据转化为NumPy数组
            if dataset == 'your_dataset_name':  # 请替换为你的数据集名称
                print(data)
    image = data
    filename = os.path.basename(h5_file)
    base_filename, ext = os.path.splitext(filename)
    # 显示图像
    plt.imshow(image, cmap='gray')
    plt.savefig(base_filename + ".png")
    plt.show()
    # 保存图像为png文件

    # 同时也可以显示图像
    # plt.show()





















