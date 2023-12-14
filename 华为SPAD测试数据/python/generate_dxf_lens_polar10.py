# -*- coding: utf-8 -*-
"""
生成超透镜文件
如缺少对应的package
使用 pip install numpy
    pip install dxfwrite
    pip install scipy
分别安装
author： xiong dun
version：20230107
"""
import numpy as np
import scipy.io as sio
from dxfwrite import DXFEngine as dxf

###########################################
##########以下根据需要修改
###########################################

M = 24960  # 透镜总共有多少个单元 M*M 透镜总有463*463个结构
unit_size_half = 0.2 #单位微米，单元结构尺寸的一半
filename = 'lens_polar10.dxf'

m = 10

uint_cell_temp =  sio.loadmat('unit_cell_all.mat')  
uint_cel= uint_cell_temp['unit_cell_all'].astype(np.float32) 

S_temp_1 = sio.loadmat('new_phase_map_far_int10.mat')   
R_far = S_temp_1['z'].astype(np.int8)

S_temp_2 = sio.loadmat('new_phase_map_near_int10.mat')   
R_near = S_temp_2['z'].astype(np.int8)


M_out = 24960 #
startX = 0   # X start place 
startY = 0  # Y start place

###########################################
##########以下不用修改
###########################################
def rotation(x,y,xita):
      complex_format = x+ 1j * y
      r = np.abs(complex_format)
      xita_0 = np.angle(complex_format)
      
      xita_out = xita_0 + xita
      
      x_new = r*np.cos(xita_out)
      y_new = r*np.sin(xita_out)

      return x_new,y_new
     
drawing = dxf.drawing(filename)
for i in range(m):
    for j in range(m):
         name_temp = 'MY_BLOCK%05d_%05d' % (i,j)
         block = dxf.block(name=name_temp)

         x_half = uint_cel[i,j,0] /2 *1e-3
         y_half = uint_cel[i,j,1] /2 *1e-3
         xita = uint_cel[i,j,2] /2 *2*np.pi / 180
    
         x1 = x_half
         y1 = y_half
        
         x2 = -x_half
         y2 = y_half
        
         x3 = -x_half
         y3 = -y_half
        
         x4 = x_half
         y4 = -y_half
        
         x1,y1 = rotation(x1,y1,xita)
         x2,y2 = rotation(x2,y2,xita)
         x3,y3 = rotation(x3,y3,xita)
         x4,y4 = rotation(x4,y4,xita)
        
    
         block.add(
                 dxf.polyline(points=[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)])
         )

         drawing.blocks.add(block)
  

for i in range(startX,startX+M_out):
    for j in range(startY,startY+M_out):
            print(i)
            if R_far[i,j] >=0 & R_near[i,j] >= 0:
               name_temp = 'MY_BLOCK%05d_%05d' % (R_near[i,j],R_far[i,j])
               block_instance = dxf.insert(
                    blockname=name_temp,
                    insert=(i * unit_size_half*2, j * unit_size_half*2)
               )
               drawing.add(block_instance)

drawing.save()  

####################################################       
        









