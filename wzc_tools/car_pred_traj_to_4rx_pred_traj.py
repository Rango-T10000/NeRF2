#fsc_data/pred_traj_data/pred_trajectory.json是gnss天线自己的pred_pos
#根据gnss天线和我的四个天线的相对位置关系的出4个天线分别的预测位置
import json
import numpy as np

# 读取原始数据
with open('fsc_data/pred_traj_data/pred_trajectory.json', 'r') as f:
    pred_base_traj = json.load(f)

pred_rx_pos = []
for i in range(len(pred_base_traj)):
    timestamp = pred_base_traj[i][0]
    x = pred_base_traj[i][1]
    y = pred_base_traj[i][2]
    z = pred_base_traj[i][3]
    
    # 计算4个天线的位置
    x1 = x - 0.095 * 3
    x2 = x - 0.095 * 2
    x3 = x - 0.095 * 1
    x4 = x - 0.095 * 0 
    
    y1 = y - 0.1
    y2 = y - 0.1
    y3 = y - 0.1
    y4 = y - 0.1
    
    z1 = z
    z2 = z
    z3 = z
    z4 = z
    
    # 组合成新的数据格式
    new_pos = [timestamp, 
               x1, y1, z1,
               x2, y2, z2,
               x3, y3, z3,
               x4, y4, z4]
    
    pred_rx_pos.append(new_pos)

# 保存结果
with open('fsc_data/pred_traj_data/pred_trajectory_4rx.json', 'w') as f:
    f.write('[\n')
    for i, item in enumerate(pred_rx_pos):
        json.dump(item, f)
        if i < len(pred_rx_pos) - 1:
            f.write(',\n')
        else:
            f.write('\n')
    f.write(']\n')
