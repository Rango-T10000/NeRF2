import json

# 定义文件路径
file_path = 'fsc_data/pred_traj_data/combined_gnss_data_pred_rx_pos.json'

# 读取 .pkl 文件
with open(file_path, 'r') as file:
    data = json.load(file)

# 打印读取的数据
print(data[0])