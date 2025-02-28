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
import json
import torch.nn.functional as F



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
    if dataset_type == "fsc":
        return
    else:
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
        elif dataset_type == "fsc":
            csi_dir = os.path.join(datadir, 'combined_gnss_data.json')

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


    def normalize_csi(self, csi): #训练的时候把input跟label都norm了，所以预测出的也是norm的结果，去跟norm后的label计算loss
        self.csi_max = torch.max(abs(csi))
        return csi / self.csi_max

    def denormalize_csi(self, csi): #推理的时候要知道预测的最终结果，那就需要denorm一下
        assert self.csi_max is not None, "Please normalize csi first"
        return csi * self.csi_max


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 1027]. The inputs for training
                    uplink: 52 (26 real; 26 imag), ray_o: 3, ray_d: 9x36x3, n_samples = n_dataset * n_bs, 即4000条数据 * 8个天线 = 32000条数据
                    [n_samples, 1027] = [n_dataset * n_bs, 52+3+3*36*9]
                    其实nerf2的网络输入需要tx的位置,但是这里没有tx的位置,所以只有uplink的csi数据

        nn_labels : tensor. [n_samples, 52]. The downlink channels csi as labels
        """
        ## NOTE! Large dataset may cause OOM?; 这里的len(self)是下面的__len__函数
        nn_inputs = torch.tensor(np.zeros((len(self), 52+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 52)), dtype=torch.float32)

        ## generate rays origin at gateways，其实这里实际是对基站的位置生成ray，做ray tracing，基站是原点
        bs_ray_o, bs_rays_d = self.gen_rays_gateways()
        bs_ray_o = rearrange(bs_ray_o, 'n g c -> n (g c)')   # [n_bs, 1, 3] --> [n_bs, 3]
        bs_rays_d = rearrange(bs_rays_d, 'n g c -> n (g c)') # [n_bs, n_rays, 3] --> [n_bs, n_rays*3]

        ## Load data
        for data_counter, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)): #这里的data_counter是数据集的索引, idx是self.dataset_index取出的内容
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


    def __len__(self): #因为这里写了这个函数,所以调用len(self)的时候就会调用这个函数
        return len(self.dataset_index) * self.n_bs


#这个是把所有的(rx,tx,rss,phase) * 4组，即4个rx对应的数据 当成一个sample的数据，最后是输入4个(rx,tx)，输出4组csi对应天线rx1, rx2, rx3, rx4
class fsc_dataset_4rx(Dataset):
    def __init__(self, datadir, indexdir, scale_worldsize=1):
        super().__init__()
        self.datadir = datadir
        self.csidata_dir = os.path.join(datadir, 'combined_gnss_data_pred_rx_pos.json')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resolution of rays
        
        # 加载JSON数据
        with open(self.csidata_dir, 'r') as file:
            self.json_data = json.load(file)
        
        #按照索引值把train或test set先取出来
        self.json_index_data = []
        for idx in self.dataset_index:
            self.json_index_data.append(self.json_data[idx])

        #-------------提取所有rx_positions-------------
        rx_positions_list = []
        for data_item in self.json_index_data:
            timestamp = list(data_item.keys())[0]
            com_data = data_item[timestamp]
            # 为每个时间戳创建一个包含4个rx位置的列表
            rx_positions_group = []
            # 获取当前时间戳下所有的rx_pos
            for i in range(4):
                if i < len(com_data) and com_data[i]['rx_pos']:  # 确保rx_pos不为空
                    rx_positions_group.append(com_data[i]['rx_pos'])
                else:
                    rx_positions_group.append([0,0,0])
            rx_positions_list.append(rx_positions_group)
        # 转换为tensor
        self.rx_pos = torch.tensor(rx_positions_list, dtype=torch.float32)  # shape: [1280,4,3];在某些时间戳下有的天线没有数据,强行补0保证数据完整

        #-------------提取所有[tx_positions,carrier_phase,rss],对于tx数量不固定使用补0补成一样的，用同纬度mask值标记的方法-------------
        tx_csi_list = []
        for data_item in self.json_index_data:
            timestamp = list(data_item.keys())[0]
            com_data = data_item[timestamp]
            # 为每个时间戳创建一个包含4个rx对应的tx_csi数据的列表
            tx_csi_timestamp_group = []
            
            # 获取当前时间戳下所有rx对应的tx_csi数据
            for i in range(4):
                tx_csi_for_one_rx = []
                if i < len(com_data) and com_data[i]['rx_pos']:  # 确保rx存在
                    for tx_csi in com_data[i]['tx']:
                        tx_csi_for_one_rx.append([tx_csi[m] for m in [1, 2, 3, 6, 7]])  # [tx_x, tx_y, tx_z, carrier_phase, rss]
                tx_csi_timestamp_group.append(tx_csi_for_one_rx)  # 即使是空列表也添加进去
            tx_csi_list.append(tx_csi_timestamp_group)

        # 找到每个rx对应的最大tx数量
        max_tx_nums = []
        for timestamp_data in tx_csi_list:  # timestamp_data是每个时间戳的数据，包含4个rx的数据
            for rx_data in timestamp_data:  # rx_data是每个rx的数据，包含多个tx的数据
                max_tx_nums.append(len(rx_data))
        self.max_tx_num = max(max_tx_nums)  #10个

        # 初始化 mask 列表并补齐数据
        tx_csi_padded_list = []
        tx_csi_mask_list = []

        for timestamp_data in tx_csi_list:
            timestamp_padded = []
            timestamp_mask = []
            
            for rx_data in timestamp_data:
                # 为每个rx创建mask
                rx_mask = [1] * len(rx_data) + [0] * (self.max_tx_num - len(rx_data))
                timestamp_mask.append(rx_mask)
                
                # 补齐每个rx的数据
                rx_padded = rx_data.copy()
                while len(rx_padded) < self.max_tx_num:
                    rx_padded.append([0, 0, 0, 0, 0])
                timestamp_padded.append(rx_padded)
            
            tx_csi_padded_list.append(timestamp_padded)
            tx_csi_mask_list.append(timestamp_mask)

        # 转换为tensor
        self.tx_csi = torch.tensor(tx_csi_padded_list, dtype=torch.float32)  # shape: [1280, 4, max_tx_num, 5]
        self.tx_csi_mask = torch.tensor(tx_csi_mask_list, dtype=torch.float32)  # shape: [1280, 4, max_tx_num]

        #注意，这里的self.tx_csi_mask是用于标记tx_csi的mask，即原本有的数据我们标记为1，补0的数据我们标记为0
        #这样在训练结束的时候我们可以根据这个mask来计算loss，不计算补0的数据的loss

        self.rx_pos = self.rx_pos.unsqueeze(2).repeat(1, 1, self.max_tx_num, 1) #[1280, 4, 3]---->[1280, 4, 10, 3]
        self.tx_pos = self.tx_csi[...,:3]   #[1280, 4, max_tx_num, 3]；[1280, 4, 10, 3]
        csi_data = self.tx_csi[...,3:] #[1280, 4, max_tx_num, 1+1]；[1280, 4, 10, 2] (carrier_phase,rss)

        # self.rx_pos = rearrange(self.rx_pos, 'n g c -> (n g) c') #[5040 * max_tx_num, 3]；[50400,3]
        # self.tx_pos = rearrange(self.tx_pos, 'n g c -> (n g) c') #[5040 * max_tx_num, 3]；[50400,3]
        # csi_data = rearrange(csi_data, 'n g c -> (n g) c') #[5040 * max_tx_num, 1+1]；[50400,2] (carrier_phase,rss)

        self.carrier_phase = csi_data[...,0:1] #[1280, 4, 10, 1]
        self.rss = csi_data[...,1:]            #[1280, 4, 10, 1]

        # self.tx_csi_mask = self.tx_csi_mask.reshape(-1) # shape: [5040 * max_tx_num]; # shape: [50400]
        # self.tx_csi_mask = self.tx_csi_mask.unsqueeze(1).repeat(1, 2) # shape: [5040 * max_tx_num, 2]; # shape: [50400,2]

        #normlize
        self.carrier_phase = self.normalize_carrier_phase(self.carrier_phase) #[1280, 4, 10, 1]
        self.rss = self.normalize_rss(self.rss)                               #[1280, 4, 10, 1]
        self.csi = torch.cat([self.rss,self.carrier_phase],dim = -1)          #[1280, 4, 10, 2] 注意这里开始把rss放在carrier_phase的前面
        self.rx_pos = self.normalize_rx_pos(self.rx_pos)  #[1280, 4, 10, 3]
        self.tx_pos = self.normalize_tx_pos(self.tx_pos)  #[1280, 4, 10, 3]

        #调整维度：[1280, 4, 10, x]---->[1280,10, 4, x]---->[1280*10, 4, x]
        self.rx_pos = self.rx_pos.permute(0, 2, 1, 3).reshape(-1, 4, 3)  # [12800, 4, 3]
        self.tx_pos = self.tx_pos.permute(0, 2, 1, 3).reshape(-1, 4, 3)  # [12800, 4, 3]
        self.csi = self.csi.permute(0, 2, 1, 3).reshape(-1, 4, 2)  # [12800, 4, 2]

        # 处理数据
        self.nn_inputs, self.nn_labels = self.load_data()
        
    def load_data(self):
        """准备训练数据"""
        # 初始化输入输出张量
        n_samples = self.rx_pos.size(0)  # 12800
        n_rx = self.rx_pos.size(1)       # 4
        input_size = 3 + 3 * self.alpha_res * self.beta_res + 3    # rx_pos, rays_d, tx_pos; 978
        output_size = 2     # carrier_phase, rss; 2
        nn_inputs = torch.zeros((n_samples, n_rx, input_size), dtype=torch.float32)
        nn_labels = torch.zeros((n_samples, n_rx, output_size), dtype=torch.float32)
        
        # 生成射线:原点+方向
        ray_o, rays_d = self.gen_rays_gateways() # [n_samples, 4, 3], [n_samples, 4, 324, 3]
        rays_d = rearrange(rays_d, 'n r g c -> n r (g c)') # [n_samples, 4, 324*3]
        
        # 为每个样本准备数据
        for idx in tqdm(range(n_samples)):
            for rx_idx in range(n_rx):
                rx_pos = self.rx_pos[idx, rx_idx]  # [3]
                tx_pos = self.tx_pos[idx, rx_idx]  # [3]
                csi = self.csi[idx, rx_idx]        # [2]
                
                nn_inputs[idx, rx_idx, :3] = rx_pos
                nn_inputs[idx, rx_idx, 3:3+3*self.alpha_res*self.beta_res] = rays_d[idx, rx_idx]
                nn_inputs[idx, rx_idx, 3+3*self.alpha_res*self.beta_res:] = tx_pos
                nn_labels[idx, rx_idx, :] = csi

        return nn_inputs, nn_labels
    

    def normalize_carrier_phase(self, carrier_phase): #训练的时候把input跟label都norm了，所以预测出的也是norm的结果，去跟norm后的label计算loss
        self.carrier_phase_max = torch.max(abs(carrier_phase))
        return carrier_phase / self.carrier_phase_max

    def denormalize_carrier_phase(self, carrier_phase): #推理的时候要知道预测的最终结果，那就需要denorm一下
        assert self.carrier_phase_max is not None, "Please normalize csi first"
        return carrier_phase * self.carrier_phase_max
    
    def normalize_rss(self, rss): #训练的时候把input跟label都norm了，所以预测出的也是norm的结果，去跟norm后的label计算loss
        self.rss_max = torch.max(abs(rss))
        return rss / self.rss_max

    def denormalize_rss(self, rss): #推理的时候要知道预测的最终结果，那就需要denorm一下
        assert self.rss_max is not None, "Please normalize csi first"
        return rss * self.rss_max
    
    def normalize_tx_pos(self, tx_pos):
        """Normalize transmitter ECEF coordinates by finding max absolute value for each dimension

        Args:
            tx_pos: tensor of shape [1280, 4, 10, 3] containing ECEF coordinates
        Returns:
            normalized positions tensor of same shape
        """
        # Find max absolute value for each dimension (x, y, z)
        self.tx_pos_max_xyz = torch.amax(torch.abs(tx_pos), dim=(0,1,2), keepdim=True)  # Shape: [1,1,1,3]

        # Normalize each dimension separately
        return tx_pos / self.tx_pos_max_xyz

    def normalize_rx_pos(self, rx_pos):
        """Normalize receiver ECEF coordinates by finding max absolute value for each dimension

        Args:
            rx_pos: tensor of shape [1280, 4, 10, 3] containing ECEF coordinates
        Returns:
            normalized positions tensor of same shape
        """
        # Find max absolute value for each dimension (x, y, z)
        self.rx_pos_max_xyz = torch.amax(torch.abs(rx_pos), dim=(0,1,2), keepdim=True)  # Shape: [1,1,1,3]

        # Normalize each dimension separately
        return rx_pos / self.rx_pos_max_xyz

    def denormalize_tx_pos(self, normalized_tx_pos):
        """Denormalize the transmitter positions back to ECEF coordinates"""
        assert hasattr(self, 'tx_pos_max_xyz'), "Please normalize tx_pos first"
        return normalized_tx_pos * self.tx_pos_max_xyz

    def denormalize_rx_pos(self, normalized_rx_pos):
        """Denormalize the receiver positions back to ECEF coordinates"""
        assert hasattr(self, 'rx_pos_max_xyz'), "Please normalize rx_pos first"
        return normalized_rx_pos * self.rx_pos_max_xyz



    def gen_rays_gateways(self):
        """生成射线采样点，考虑4个rx"""
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)
        betas = betas.repeat_interleave(self.alpha_res)
        
        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)
        
        r_d = torch.stack([x, y, z], axis=0).T  # [324, 3]
        # 扩展维度以包含样本数量和rx数量
        r_d = r_d.expand([self.rx_pos.size(0), 4, self.beta_res * self.alpha_res, 3])  # [n_samples, 4, 324, 3]
        r_o = self.rx_pos  # [n_samples, 4, 3]
        
        return r_o.contiguous(), r_d.contiguous()
    

    def __getitem__(self, index):
        """修改__getitem__方法来返回包含4个rx的数据"""
        inputs = self.nn_inputs[index]    # [4, 978]
        labels = self.nn_labels[index]    # [4, 2]
        mask = self.tx_csi_mask[index]    # [4, max_tx_num]
        return inputs, labels, mask
    

    def __len__(self):  #因为这里写了这个函数,所以调用len(self)的时候就会调用这个函数
        return self.rx_pos.size(0) # 返回12800


#这个是把所有的(rx,tx,rss,phase)组当成一个sample的数据，最后是输入一个(rx,tx)，输出一组csi
class fsc_dataset(Dataset):
    def __init__(self, datadir, indexdir, scale_worldsize=1):
        super().__init__()
        self.datadir = datadir
        self.csidata_dir = os.path.join(datadir, 'combined_gnss_data_pred_rx_pos.json')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resolution of rays
        
        # 加载JSON数据
        with open(self.csidata_dir, 'r') as file:
            self.json_data = json.load(file)
        
        #按照索引值把train或test set先取出来
        self.json_index_data = []
        for idx in self.dataset_index:
            self.json_index_data.append(self.json_data[idx])

        #-------------提取所有rx_positions-------------
        rx_positions_list = []
        for data_item in self.json_index_data:
            timestamp = list(data_item.keys())[0]
            com_data = data_item[timestamp]
            # 获取当前时间戳下所有的rx_pos
            for i in range(4):
                if i < len(com_data) and com_data[i]['rx_pos']:  # 确保tx_pos不为空
                    rx_positions_list.append(com_data[i]['rx_pos'])
                else:
                    rx_positions_list.append([0,0,0])
        # 转换为tensor
        self.rx_pos = torch.tensor(rx_positions_list, dtype=torch.float32)  # shape: [1260*4,3]=[5040,3];理论是应该len = samples * 4个天线，但是在某些时间戳下有的天线没有数据,强行补0保证数据完整

        #-------------提取所有[tx_positions,carrier_phase,rss],对于tx数量不固定使用补0补成一样的，用同纬度mask值标记的方法-------------
        tx_csi_list = []
        for data_item in self.json_index_data:
            timestamp = list(data_item.keys())[0]
            com_data = data_item[timestamp]
            # 获取当前时间戳下所有的tx_pos+rss+carrier_phase
            for i in range(4):
                tx_csi_for_one_rx_list = []
                if i < len(com_data) and com_data[i]['rx_pos']:  # 确保tx_pos不为空
                    for tx_csi in com_data[i]['tx']:
                        tx_csi_for_one_rx_list.append([tx_csi[m] for m in [1, 2, 3, 6, 7]])# [tx_x, tx_y, tx_z, carrier_phase, rss]
                else:
                    tx_csi_for_one_rx_list.append([])  #对于一个timestamp下不满足4个rx对应数据的，就给缺的rx对应的tx数据补空白？,即只收到一个tx，其
                tx_csi_list.append(tx_csi_for_one_rx_list)
        
        #对于某一秒中不够4个rx对应的数的，在tx_csi_list列表中使其对应的从[[]]-->[],这样的话对应的长度就是0,下面的mask才能算对
        for n in range(len(tx_csi_list)):
            if tx_csi_list[n][0] == []:
                tx_csi_list[n] =[]
                
        # 找到第二层列表的最大长度和最小长度
        self.max_tx_num = max(len(sublist) for sublist in tx_csi_list) #10
        self.min_tx_num = min(len(sublist) for sublist in tx_csi_list) #0
        # 初始化 mask 列表
        tx_csi_mask_list = []
        # 补齐不够的部分并生成 mask
        for sublist in tx_csi_list:
            mask = [1] * len(sublist)
            while len(sublist) < self.max_tx_num:
                sublist.append([0, 0, 0, 0, 0])  # 补齐0
                mask.append(0)
            tx_csi_mask_list.append(mask)
        # 转换为 tensor
        self.tx_csi = torch.tensor(tx_csi_list, dtype=torch.float32)  # shape: [5040, max_tx_num, 3+1+1]；[5040,10,5]
        self.tx_csi_mask = torch.tensor(tx_csi_mask_list, dtype=torch.float32)  # shape: [5040, max_tx_num] 用于标记tx_csi的mask，即哪些是真实数据，哪些是补0的数据

        #注意，这里的self.tx_csi_mask是用于标记tx_csi的mask，即原本有的数据我们标记为1，补0的数据我们标记为0
        #这样在训练结束的时候我们可以根据这个mask来计算loss，不计算补0的数据的loss

        self.rx_pos = self.rx_pos.unsqueeze(1).repeat(1, self.max_tx_num, 1) #[5040, max_tx_num, 3]；[5040,10,3]
        self.tx_pos = self.tx_csi[...,:3]   #[5040, max_tx_num, 3]；[5040,10,3]
        csi_data = self.tx_csi[...,3:] #[5040, max_tx_num, 1+1]；[5040,10,2] (carrier_phase,rss)
        self.rx_pos = rearrange(self.rx_pos, 'n g c -> (n g) c') #[5040 * max_tx_num, 3]；[50400,3]
        self.tx_pos = rearrange(self.tx_pos, 'n g c -> (n g) c') #[5040 * max_tx_num, 3]；[50400,3]
        csi_data = rearrange(csi_data, 'n g c -> (n g) c') #[5040 * max_tx_num, 1+1]；[50400,2] (carrier_phase,rss)
        self.carrier_phase = csi_data[:,0:1]
        self.rss = csi_data[:,1:]
        self.tx_csi_mask = self.tx_csi_mask.reshape(-1) # shape: [5040 * max_tx_num]; # shape: [50400]
        self.tx_csi_mask = self.tx_csi_mask.unsqueeze(1).repeat(1, 2) # shape: [5040 * max_tx_num, 2]; # shape: [50400,2]

        #normlize
        self.carrier_phase = self.normalize_carrier_phase(self.carrier_phase) #[5040 * max_tx_num, 1]；[50400,1] 
        self.rss = self.normalize_rss(self.rss)                               #[5040 * max_tx_num, 1]；[50400,1] 
        self.csi = torch.cat([self.rss,self.carrier_phase],dim = -1)          #[5040 * max_tx_num, 2]；[50400,2]
        self.rx_pos = self.normalize_rx_pos(self.rx_pos)
        self.tx_pos = self.normalize_tx_pos(self.tx_pos)

        # 处理数据
        self.nn_inputs, self.nn_labels = self.load_data()
        
    def load_data(self):
        """准备训练数据"""
        # # 初始化输入输出张量
        n_samples = self.rx_pos.size(0) #50400
        input_size = 3 + 3 * self.alpha_res * self.beta_res + 3    # rx_pos, rays_d, tx_pos; 978
        output_size = 2     # carrier_phase, rss; 2
        nn_inputs = torch.zeros((n_samples, input_size), dtype=torch.float32) # rx_pos, rays_d, tx_pos
        nn_labels = torch.zeros((n_samples, output_size), dtype=torch.float32)# carrier_phase, rss
        
        # 生成射线:原点+方向
        ray_o, rays_d = self.gen_rays_gateways() # [n_samples, 3], [n_samples, 36*9, 3]; [50400,324,3]
        rays_d = rearrange(rays_d, 'n g c -> n (g c)') # [n_samples, 36*9*3]; [50400,324*3]
        
        # 为每个样本准备数据
        for data_counter, idx in tqdm(enumerate(range(n_samples)), total=n_samples): 
            rx_pos = self.rx_pos[idx]  # [3]
            tx_pos = self.tx_pos[idx]  # [3]
            csi = self.csi[idx]  # [2]
            nn_inputs[idx, :3] = rx_pos
            nn_inputs[idx, 3:3+3*self.alpha_res*self.beta_res] = rays_d[idx]
            nn_inputs[idx, 3+3*self.alpha_res*self.beta_res:] = tx_pos # tx_pos
            nn_labels[idx, :] = csi #csi

        return nn_inputs, nn_labels
    

    def normalize_carrier_phase(self, carrier_phase): #训练的时候把input跟label都norm了，所以预测出的也是norm的结果，去跟norm后的label计算loss
        self.carrier_phase_max = torch.max(abs(carrier_phase))
        return carrier_phase / self.carrier_phase_max

    def denormalize_carrier_phase(self, carrier_phase): #推理的时候要知道预测的最终结果，那就需要denorm一下
        assert self.carrier_phase_max is not None, "Please normalize csi first"
        return carrier_phase * self.carrier_phase_max
    
    def normalize_rss(self, rss): #训练的时候把input跟label都norm了，所以预测出的也是norm的结果，去跟norm后的label计算loss
        self.rss_max = torch.max(abs(rss))
        return rss / self.rss_max

    def denormalize_rss(self, rss): #推理的时候要知道预测的最终结果，那就需要denorm一下
        assert self.rss_max is not None, "Please normalize csi first"
        return rss * self.rss_max
    
    def normalize_tx_pos(self, tx_pos):
        """Normalize transmitter ECEF coordinates by finding max absolute value for each dimension
        
        Args:
            tx_pos: tensor of shape [..., 3] containing ECEF coordinates
        Returns:
            normalized positions tensor of same shape
        """
        # Find max absolute value for each dimension (x,y,z)
        self.tx_pos_max_xyz = torch.max(torch.abs(tx_pos), dim=0)[0]  # [3]
        
        # Normalize each dimension separately
        return tx_pos / self.tx_pos_max_xyz

    def normalize_rx_pos(self, rx_pos):
        """Normalize receiver ECEF coordinates by finding max absolute value for each dimension
        
        Args:
            rx_pos: tensor of shape [..., 3] containing ECEF coordinates
        Returns:
            normalized positions tensor of same shape
        """
        # Find max absolute value for each dimension (x,y,z)
        self.rx_pos_max_xyz = torch.max(torch.abs(rx_pos), dim=0)[0]  # [3]
        
        # Normalize each dimension separately
        return rx_pos / self.rx_pos_max_xyz

    def denormalize_tx_pos(self, normalized_tx_pos):
        """Denormalize the transmitter positions back to ECEF coordinates"""
        assert hasattr(self, 'tx_pos_max_xyz'), "Please normalize tx_pos first"
        return normalized_tx_pos * self.tx_pos_max_xyz

    def denormalize_rx_pos(self, normalized_rx_pos):
        """Denormalize the receiver positions back to ECEF coordinates"""
        assert hasattr(self, 'rx_pos_max_xyz'), "Please normalize rx_pos first"
        return normalized_rx_pos * self.rx_pos_max_xyz

    def gen_rays_gateways(self):
        """生成射线采样点，与CSI_dataset相同"""
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)
        betas = betas.repeat_interleave(self.alpha_res)
        
        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)
        
        r_d = torch.stack([x, y, z], axis=0).T # [9*36, 3]
        r_d = r_d.expand([self.rx_pos.size(0), self.beta_res * self.alpha_res, 3]) # [n_samples, 9*36, 3]
        r_o = self.rx_pos  # 使用接收机位置作为射线原点 [n_samples, 3]
        
        return r_o.contiguous(), r_d.contiguous()
    

    def __getitem__(self, index):
        """修改__getitem__方法来同时返回mask"""
        inputs = self.nn_inputs[index]
        labels = self.nn_labels[index]
        mask = self.tx_csi_mask[index]  # [max_tx_num] #拓展两倍！
        return inputs, labels, mask
    

    def __len__(self):  #因为这里写了这个函数,所以调用len(self)的时候就会调用这个函数
        return self.rx_pos.size(0)




dataset_dict = {"rfid": Spectrum_dataset, "ble": BLE_dataset, "mimo": CSI_dataset, "fsc": fsc_dataset, "fsc_4rx": fsc_dataset_4rx}
