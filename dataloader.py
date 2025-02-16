# -*- coding: utf-8 -*-
"""dataset processing and loading
"""
import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange



# def rssi2amplitude(rssi):
#     """convert rssi to amplitude
#     """
#     return 100 * 10 ** (rssi / 20)


# def amplitude2rssi(amplitude):
#     """convert amplitude to rssi
#     """
#     return 20 * np.log10(amplitude / 100)


def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


def split_dataset(datadir, ratio=0.8, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)
    elif dataset_type == "ble":
        rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        index = pd.read_csv(rssi_dir).index.values
        random.shuffle(index)
    elif dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.npy')
        index = [i for i in range(np.load(csi_dir).shape[0])]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')




class Spectrum_dataset(Dataset):
    """spectrum dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1) -> None:
        super().__init__()
        self.datadir = datadir
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_info.yml')
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])
        example_spt = imageio.imread(os.path.join(self.spectrum_dir, self.spt_names[0]))
        self.n_elevation, self.n_azimuth = example_spt.shape
        self.rays_per_spectrum = self.n_elevation * self.n_azimuth
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        self.nn_inputs, self.nn_labels = self.load_data()


    def __len__(self):
        return len(self.dataset_index) * self.rays_per_spectrum


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        train_inputs : tensor. [n_samples, 9]. The inputs for training ray_o, ray_d, tx_pos (3 ,3, 3)
                    即后面的nn_inputs: [n_samples, 9], e.g.[90 * 360 * 6123个空间谱图片 = 158695200 , 9] = [158695200, 9]
                    即输入就是实际一个RX阵列(gateway)接收信号,以这个RX阵列的位置和方向算出来的ray,以及tx的位置

        train_labels : tensor. [n_samples, 1]. The RSSI labels for training
                    即后面的nn_labels: [n_samples, 1], e.g.[90 * 360 * 6123个空间谱图片 = 158695200 , 1] = [158695200, 1]
                    即输出就是这个RX阵列接收到的信号强度,即空间谱图片的每个pixel的值,这个仅仅反映了rss,没有相位信息
        """
        ## NOTE! Each spectrum will cost 1.2 MB of memory. Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 9)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        ## Load gateway position and orientation; 这里说的gateway是指实际的RX阵列，训练的时候给出这个,用于下面计算ray_o (以RX为起点)和ray_d (ray的方向)
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_info = yaml.safe_load(f)
            gateway_pos = gateway_info['gateway1']['position']
            gateway_orientation = gateway_info['gateway1']['orientation']

        ## Load transmitter position
        tx_pos = pd.read_csv(self.tx_pos_dir).values
        tx_pos = torch.tensor(tx_pos, dtype=torch.float32) # [n_samples, 3],e.g.[6123, 3],实际tx的位置变了6123次采的数据

        ## Load data, each spectrum contains 90x360 pixels(rays)
        for i, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            spectrum = imageio.imread(os.path.join(self.spectrum_dir, idx + '.png')) / 255 # 安装图片id读取对应的空间谱图片
            spectrum = torch.tensor(spectrum, dtype=torch.float32).view(-1, 1) # [n_rays, 1],空间谱图片每个pixel就是一个ray，所以这里是[360 * 90, 1]
            ray_o, ray_d = self.gen_rays_spectrum(gateway_pos, gateway_orientation) #ray的起点跟方向是基于我RX阵列的位置和方向计算的
            tx_pos_i = torch.tile(tx_pos[int(idx)-1], (self.rays_per_spectrum,)).reshape(-1,3)  # [n_rays, 3]
            nn_inputs[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :9] = \
                torch.cat([ray_o, ray_d, tx_pos_i], dim=1)
            nn_labels[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :] = spectrum

        return nn_inputs, nn_labels


    def gen_rays_spectrum(self, gateway_pos, gateway_orientation):
        """generate sample rays origin at gateway with resolution given by spectrum

        Parameters
        ----------
        azimuth : int. The number of azimuth angles
        elevation : int. The number of elevation angles

        Returns
        -------
        r_o : tensor. [n_rays, 3]. The origin of rays
        r_d : tensor. [n_rays, 3]. The direction of rays, unit vector
        """

        azimuth = torch.linspace(1, 360, self.n_azimuth) / 180 * np.pi      # [1,2,3...360]共360个
        elevation = torch.linspace(1, 90, self.n_elevation) / 180 * np.pi   # [1,2,3...90]共90个
        azimuth = torch.tile(azimuth, (self.n_elevation,))  # [1,2,3...360,1,2,3...360,...] pytorch 2.0
        elevation = torch.repeat_interleave(elevation, self.n_azimuth)  # [1,1,1,...,2,2,2,...,90,90,90,...]

        x = 1 * torch.cos(elevation) * torch.cos(azimuth) # [n_azi * n_ele], i.e., [n_rays]
        y = 1 * torch.cos(elevation) * torch.sin(azimuth)
        z = 1 * torch.sin(elevation)

        r_d = torch.stack([x, y, z], dim=0)  # [3, n_rays] 3D direction of rays in gateway coordinate
        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        r_d = R @ r_d  # [3, n_rays] 3D direction of rays in world coordinate
        gateway_pos = torch.tensor(gateway_pos, dtype=torch.float32)
        r_o = torch.tile(gateway_pos, (self.rays_per_spectrum,)).reshape(-1, 3)  # [n_rays, 3] #ray的起点是RX阵列/gateway的位置

        return r_o, r_d.T





class BLE_dataset(Dataset):
    """ble dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1) -> None:
        super().__init__()
        self.datadir = datadir
        tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_position.yml')
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load gateway position
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_pos_dict = yaml.safe_load(f)
            self.gateway_pos = torch.tensor([pos for pos in gateway_pos_dict.values()], dtype=torch.float32)
            self.gateway_pos = self.gateway_pos / scale_worldsize
            self.n_gateways = len(self.gateway_pos)

        # Load transmitter position
        self.tx_poses = torch.tensor(pd.read_csv(tx_pos_dir).values, dtype=torch.float32)
        self.tx_poses = self.tx_poses / scale_worldsize

        # Load gateway received RSSI
        self.rssis = torch.tensor(pd.read_csv(self.rssi_dir).values, dtype=torch.float32)

        self.nn_inputs, self.nn_labels = self.load_data()


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        nn_inputs : tensor. [n_samples, 978]. The inputs for training
                    tx_pos:3, ray_o:3, ray_d:9x36x3,
        nn_labels : tensor. [n_samples, 1]. The RSSI labels for training
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        ## generate rays origin at gateways
        gateways_ray_o, gateways_rays_d = self.gen_rays_gateways()

        ## Load data
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):
            rssis = self.rssis[idx]
            tx_pos = self.tx_poses[idx].view(-1)  # [3]
            for i_gateway, rssi in enumerate(rssis):
                if rssi != -100:
                    gateway_ray_o = gateways_ray_o[i_gateway].view(-1)  # [3]
                    gateway_rays_d = gateways_rays_d[i_gateway].view(-1)  # [n_rays x 3]
                    nn_inputs[data_counter] = torch.cat([tx_pos, gateway_ray_o, gateway_rays_d], dim=-1)
                    nn_labels[data_counter] = rssi
                    data_counter += 1

        nn_labels = rssi2amplitude(nn_labels)

        return nn_inputs, nn_labels


    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_gateways, 1, 3]. The origin of rays
        r_d : tensor. [n_gateways, n_rays, 3]. The direction of rays, unit vector
        """


        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.n_gateways, self.beta_res * self.alpha_res, 3])  # [n_gateways, 9*36, 3]
        r_o = self.gateway_pos.unsqueeze(1) # [21, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d


    def __len__(self):
        rssis = self.rssis[self.dataset_index]
        return torch.sum(rssis != -100)

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]




class CSI_dataset(Dataset):

    def __init__(self, datadir, indexdir, scale_worldsize=1):
        """ datasets [datalen*8, up+down+r_o+r_d] --> [datalen*8, 26+26+3+36*3]
        """
        super().__init__()
        self.datadir = datadir
        self.csidata_dir = os.path.join(datadir, 'csidata.npy')
        self.bs_pos_dir = os.path.join(datadir, 'base-station.yml')
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load base station position,即tx的位置
        with open(os.path.join(self.bs_pos_dir)) as f:
            bs_pos_dict = yaml.safe_load(f)
            self.bs_pos = torch.tensor([bs_pos_dict["base_station"]], dtype=torch.float32).squeeze()
            self.bs_pos = self.bs_pos / scale_worldsize
            self.n_bs = len(self.bs_pos)

        # load CSI data
        # 因为数据集是8-antenna base station收到的信号的CSI数据，而每个天线有52个subcarriers的CSI数据，取前26个作为uplink的csi数据，后26个作为downlink的csi数据，这样用上行预测下行
        csi_data = torch.from_numpy(np.load(self.csidata_dir))  #[N, 8, 52] 这里的8是8个天线，每个天线有52个subcarriers的CSI数据
        csi_data = self.normalize_csi(csi_data) #稳定训练，防止梯度爆炸，确保所有 CSI 值的范围在 [-1, 1] 之间
        uplink, downlink = csi_data[..., :26], csi_data[..., 26:]
        up_real, up_imag = torch.real(uplink), torch.imag(uplink)
        down_real, down_imag = torch.real(downlink), torch.imag(downlink)
        self.uplink = torch.cat([up_real, up_imag], dim=-1)    # [N, 8, 52]
        self.downlink = torch.cat([down_real, down_imag], dim=-1)    # [N, 8, 52]
        self.uplink = rearrange(self.uplink, 'n g c -> (n g) c')    # [N*8, 52]
        self.downlink = rearrange(self.downlink, 'n g c -> (n g) c')    # [N*8, 52]

        self.nn_inputs, self.nn_labels = self.load_data()


    def normalize_csi(self, csi):
        self.csi_max = torch.max(abs(csi))
        return csi / self.csi_max

    def denormalize_csi(self, csi):
        assert self.csi_max is not None, "Please normalize csi first"
        return csi * self.csi_max


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 1027]. The inputs for training
                    uplink: 52 (26 real; 26 imag), ray_o: 3, ray_d: 9x36x3, n_samples = n_dataset * n_bs
                    [n_samples, 1027] = [n_dataset * n_bs, 52+3+3*36*9]
                    其实nerf2的网络输入需要tx的位置,但是这里没有tx的位置,所以只有uplink的csi数据

        nn_labels : tensor. [n_samples, 52]. The downlink channels csi as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 52+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 52)), dtype=torch.float32)

        ## generate rays origin at gateways，其实这里实际是对基站的位置生成ray，做ray tracing，基站是原点
        bs_ray_o, bs_rays_d = self.gen_rays_gateways()
        bs_ray_o = rearrange(bs_ray_o, 'n g c -> n (g c)')   # [n_bs, 1, 3] --> [n_bs, 3]
        bs_rays_d = rearrange(bs_rays_d, 'n g c -> n (g c)') # [n_bs, n_rays, 3] --> [n_bs, n_rays*3]

        ## Load data
        for data_counter, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            bs_uplink = self.uplink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            bs_downlink = self.downlink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            nn_inputs[data_counter*self.n_bs: (data_counter+1)*self.n_bs] = torch.cat([bs_uplink, bs_ray_o, bs_rays_d], dim=-1) # [n_bs, 52+3+3*36*9]
            nn_labels[data_counter*self.n_bs: (data_counter+1)*self.n_bs]  = bs_downlink
        return nn_inputs, nn_labels

    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_bs, 1, 3]. The origin of rays
        r_d : tensor. [n_bs, n_rays, 3]. The direction of rays, unit vector
        """
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi  #[0, 10, 20, ..., 350]从0度开始
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi    #[10, 20, ..., 90]从10度开始
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (9*36)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3] ；.T即.transpose(0, 1)交换下纬度
        r_d = r_d.expand([self.n_bs, self.beta_res * self.alpha_res, 3])  # [n_bs, 9*36, 3]，即每个基站有9*36个ray
        r_o = self.bs_pos.unsqueeze(1) # [n_bs, 1, 3]  #bs基站的位置就是原点
        r_o, r_d = r_o.contiguous(), r_d.contiguous() #确保张量在内存中是连续的，以便后续计算可以顺利进行，并提高计算效率。

        return r_o, r_d


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def __len__(self):
        return len(self.dataset_index) * self.n_bs


dataset_dict = {"rfid": Spectrum_dataset, "ble": BLE_dataset, "mimo": CSI_dataset}
