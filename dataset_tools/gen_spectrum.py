# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sconst
import torch
import imageio.v2 as imageio

# x,y,z coordinates of 16 antennas, customized for your own antenna array
# ANT_LOC = [[-0.24, -0.24, 0], [-0.08, -0.24, 0], [0.08, -0.24, 0], [0.24, -0.24, 0],
#            [-0.24, -0.08, 0], [-0.08, -0.08, 0], [0.08, -0.08, 0], [0.24, -0.08, 0],
#            [-0.24,  0.08, 0], [-0.08,  0.08, 0], [0.08,  0.08, 0], [0.24,  0.08, 0],
#            [-0.24,  0.24, 0], [-0.08,  0.24, 0], [0.08,  0.24, 0], [0.24,  0.24, 0]]

# 修改后的线阵坐标（沿x轴排列，间隔0.095m）
ANT_LOC = [[-0.1425, 0, 0], [-0.0475, 0, 0], [0.0475, 0, 0], [0.1425, 0, 0]]

normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

class Bartlett():
    """ Class to generate Spatial Spectrum using Bartlett Algorithm. """
    """ Modified for linear array (1x4) """
    def __init__(self, frequency=1.575e9):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.antenna_loc = torch.tensor(ANT_LOC, dtype=torch.float32).T  # 3x4
        self.lamda = sconst.c / frequency
        self.theory_phase = self._calculate_theory_phase_linear().to(self.device)

    def _calculate_theory_phase(self): #这个是计算的各个阵子之间的理论相位差；是对所有可能的方向360*90的方向来的信号计算各阵子收到信号的理论相位差
        """ Calculates theoretical phase difference over both azimuthal and elevation angle. """
        azimuth = torch.linspace(0, 359, 360) / 180 * np.pi
        elevation = torch.linspace(1, 90, 90) / 180 * np.pi

        # azimuth[0,1,..0,1..], elevation [0,0,..1,1..]
        elevation_grid, azimuth_grid = torch.meshgrid(elevation, azimuth, indexing="ij")
        azimuth_grid = azimuth_grid.flatten()
        elevation_grid = elevation_grid.flatten()

        theory_dis_diff = (self.antenna_loc[0,:].unsqueeze(-1) * torch.cos(azimuth_grid) * torch.cos(elevation_grid) +
                        self.antenna_loc[1,:].unsqueeze(-1) * torch.sin(azimuth_grid) * torch.cos(elevation_grid))
        theory_phase = -2 * np.pi * theory_dis_diff / self.lamda
        return theory_phase.T

    def _calculate_theory_phase_linear(self):
        """ 对于linear array, 仅计算方位角理论相位差（0-180度） """
        azimuth = torch.linspace(0, 180, 1810) / 180 * np.pi  # 0.1度分辨率
        
        # 提取天线x坐标 (形状: 4)
        x_coords = self.antenna_loc[0, :]
        
        # 理论波程差计算 (形状: 4x181)
        theory_dis_diff = x_coords.unsqueeze(-1) * torch.cos(azimuth)
        
        # 转换为相位差 (形状: 181x4)
        theory_phase = -2 * np.pi * theory_dis_diff / self.lamda
        return theory_phase.T  


    def gen_spectrum(self, phase_measurements):
        """ Generates spatial spectrum from phase measurements. """
        phase_measurements = torch.tensor(phase_measurements, dtype=torch.float32).to(self.device)
        delta_phase = self.theory_phase - phase_measurements.reshape(1, -1)   # (360x90,16) - 1x16；计算理论相位和测量相位之间的差值
        #如下两步是根据理论与测量之间差值计算各个方向360*90接收信号相对强度，即空间谱
        phase_sum = torch.exp(1j * delta_phase).sum(1) / self.antenna_loc.shape[1] 
        spectrum = normalize(torch.abs(phase_sum)).view(90, 360).cpu().numpy()
        return spectrum

    def gen_spectrum_linear(self, phase_measurements):
        """对于linear array 生成空间谱 """
        phase_measurements = torch.tensor(phase_measurements, dtype=torch.float32).to(self.device)
        delta_phase = self.theory_phase - phase_measurements.reshape(1, -1)  # (181,4) - (1,4)
        phase_sum = torch.exp(1j * delta_phase).sum(1) / self.antenna_loc.shape[1]
        spectrum = normalize(torch.abs(phase_sum)).cpu().numpy()
        
        # 转换为二维图像格式 (高度扩展为100像素以便显示)，其实只有一个强度值，就是复制100份方便显示
        spectrum = spectrum.reshape(-1, 1)
        spectrum = np.repeat(spectrum, 100, axis=1)
        return (spectrum * 255).astype(np.uint8)


if __name__ == '__main__':

    # #这个是你实际测量到的相位，以第一个个针子为基准，就可以计算其他阵子相对于第一个阵子的相位差
    # sample_phase = [-1.886,-1.923,-2.832,-1.743,
    #             -1.751,-1.899,-2.370,-3.113,
    #             -2.394,-2.464,2.964,-2.904,
    #             -1.573,-2.525,-3.039,-2.839]
    # worker = Bartlett()
    # spectrum = worker.gen_spectrum(sample_phase)
    # spectrum = (spectrum * 255).astype(np.uint8)

    # imageio.imsave('dataset_tools/spectrum.png', spectrum)

    #对于linear array
    # 示例相位测量值（需要4个元素）
    sample_phase = [-1.2, -0.8, 0.3, 1.5]  # 替换为你的实际测量值
    
    worker = Bartlett()
    spectrum = worker.gen_spectrum_linear(sample_phase)
    imageio.imsave('dataset_tools/linear_spectrum.png', spectrum)