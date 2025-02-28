import pandas as pd
import numpy as np
from collections import defaultdict
import json

def read_gnss_data(file_path):
    """读取GNSS数据文件"""
    return pd.read_csv(file_path)

def process_single_timestamp_data(row_group):
    """处理单个时间戳下的数据"""
    rx_pos = [row_group['rx_x'].iloc[0], row_group['rx_y'].iloc[0], row_group['rx_z'].iloc[0]]
    
    # 处理卫星数据
    tx_data = []
    for _, row in row_group.iterrows():
        tx_data.append([
            row['prn'],
            row['sat_x'],
            row['sat_y'],
            row['sat_z'],
            row['elevation_angles'],
            row['azimuth_angles'],
            row['carrier_phase'],
            row['signal_strength']
        ])
    
    return {
        'rx_pos': rx_pos,
        'tx': tx_data
    }

def combine_gnss_data(com_files):
    """合并多个COM口的GNSS数据，按照指定时间范围每秒生成数据"""
    # 创建时间范围
    start_time = "2025-01-22 03:16:19"
    end_time = "2025-01-22 03:39:48"
    time_range = pd.date_range(start=start_time, end=end_time, freq='S')
    
    # 初始化结果字典，对每个时间戳都预设空列表
    combined_data = {str(t): [] for t in time_range}
    
    # 读取并处理每个COM口的数据
    for com_file in com_files:
        df = read_gnss_data(com_file)
        
        # 确保timestamp列是字符串类型
        df['timestamp'] = df['timestamp'].astype(str)
        
        # 按时间戳分组处理数据
        for timestamp, group in df.groupby('timestamp'):
            # 只处理在指定时间范围内的数据
            if timestamp in combined_data:
                data = process_single_timestamp_data(group)
                combined_data[timestamp].append(data)
    
    # 将字典转换为列表格式，保持时间顺序
    return [{timestamp: data} for timestamp in time_range.astype(str) for data in [combined_data[timestamp]]]

def main():
    # COM文件路径
    com_files = [
        'fsc_data/csv_align_timestamp/output_COM8.csv',
        'fsc_data/csv_align_timestamp/output_COM9.csv',
        'fsc_data/csv_align_timestamp/output_COM10.csv',
        'fsc_data/csv_align_timestamp/output_COM11.csv'
    ]
    
    # 处理数据
    result = combine_gnss_data(com_files)
    
    #保存为JSON文件
    json_path = 'fsc_data/combined_gnss_data2.json'    
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
