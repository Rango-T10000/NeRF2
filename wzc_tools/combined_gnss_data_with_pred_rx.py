#fsc_data/combined_gnss_data.json是我的原始数据集，其中的rx_pos都是真实收到的，即gt_traj
#fsc_data/pred_traj_data/pred_trajectory_4rx.json记录的是预测的对应时刻的rx_pos
#该脚本负责读取这两个.json文件，把combined_gnss_data.json中对应的rx_pos替换为预测的

import json
import numpy as np
import copy

# 读取原始数据
with open('fsc_data/pred_traj_data/pred_trajectory_4rx.json', 'r') as f:
    pred_tx_traj = json.load(f)

# 读取原始数据
with open('fsc_data/combined_gnss_data.json', 'r') as f:
    combined_gnss_data = json.load(f)

combined_gnss_data_pred_rx_pos = copy.deepcopy(combined_gnss_data)
for i in range(len(combined_gnss_data)):
    timestamp = list(combined_gnss_data_pred_rx_pos[i].keys())[0]
    value = combined_gnss_data_pred_rx_pos[i][timestamp]
    for j in range(len(value)):
        # 计算对应接收机在pred_tx_traj中的索引位置
        rx_idx = j * 3 + 1  # 3是因为每个接收机占用x,y,z三个位置
        value[j]['rx_pos'][0] = pred_tx_traj[i][rx_idx]      # x坐标
        value[j]['rx_pos'][1] = pred_tx_traj[i][rx_idx + 1]  # y坐标


# 保存结果
with open('fsc_data/pred_traj_data/combined_gnss_data_pred_rx_pos.json', 'w') as f:
    f.write('[\n')
    for i, item in enumerate(combined_gnss_data_pred_rx_pos):
        json.dump(item, f)
        if i < len(combined_gnss_data_pred_rx_pos) - 1:
            f.write(',\n')
        else:
            f.write('\n')
    f.write(']\n')
