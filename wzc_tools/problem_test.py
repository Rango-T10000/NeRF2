import torch
import numpy as np

# n_azimuth = 360
# n_elevation = 90

# azimuth = torch.linspace(1, 360, n_azimuth) / 180 * np.pi
# elevation = torch.linspace(1, 90, n_elevation) / 180 * np.pi
# azimuth = torch.tile(azimuth, (n_elevation,))  # [1,2,3...360,1,2,3...360,...] pytorch 2.0
# elevation = torch.repeat_interleave(elevation, n_azimuth)  # [1,1,1,...,2,2,2,...,90,90,90,...]

# x = 1 * torch.cos(elevation) * torch.cos(azimuth) # [n_azi * n_ele], i.e., [n_rays]
# y = 1 * torch.cos(elevation) * torch.sin(azimuth)
# z = 1 * torch.sin(elevation)

# print(x.shape)

# #x = 1 * torch.cos(elevation) * torch.cos(azimuth)如果这句无报错强行退出那就是pytorch版本不对

# #Python 的切片规则是“包含 start，但不包含 end”：主要是为了方便计算切片的长度：end - start 刚好等于切片的元素个数。
# dist = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(dist[1])
# print(dist[1:])
# print(dist[:-1])
# print(dist[1:] - dist[:-1])

tx_pos = torch.tensor(([1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4]), dtype=torch.float32)
print(tx_pos.shape)
# tx = tx_pos.reshape(-1)
tx = tx_pos.view(-1)
print(tx.shape)
print(tx)
