# #2025-01-22 03:15:11,-2416092.1683545797,5386835.186216177,2405469.698737485
# #2025-01-22 03:15:12,-2416092.2782852473,5386835.347384429,2405469.7575625717

# import numpy as np

# # 给定的ECEF坐标
# point1 = np.array([-2417911.353344458,5386123.8295825925,2405223.355866774])
# point2 = np.array([-2417911.459642158,5386123.898632818,2405223.3596622488])

# # 计算两点之间的欧几里得距离（即相对位移）
# displacement = np.linalg.norm(point2 - point1)

# print(f"相对位移: {displacement} 米")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_fsc_data(file_path):
    # 读取所有行
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 获取列名（第一行）
    headers = lines[0].strip().split()
    
    # 跳过前两行，处理数据
    data = []
    for line in lines[2:]:
        data.append(line.strip().split())
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    # 将字符串转换为浮点数
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    # 筛选指定时间段的数据
    mask = (df['UTCTime'] >= 1737515795.79) & (df['UTCTime'] <= 1737517232.98)
    df = df[mask]
    
    print("Available columns:", df.columns.tolist())
    print(f"Selected data points: {len(df)}")
    return df

# 计算合速度
def calculate_velocity(df):
    # 使用ECEF速度分量计算合速度
    velocity = np.sqrt(df['VX-ECEF']**2 + df['VY-ECEF']**2 + df['VZ-ECEF']**2)
    return velocity

def plot_velocity_time():
    # 读取数据
    df = read_fsc_data('fsc_data/20251222_2_100hz.txt')
    
    # 计算速度
    velocity = calculate_velocity(df)
    
    # 计算相对时间（从第一个时间点开始）
    time = df['UTCTime'] - df['UTCTime'].iloc[0]
    
    # 创建包含时间和速度的DataFrame
    result_df = pd.DataFrame({
        'Time(s)': time,
        'Speed(m/s)': velocity
    })
    
    # 保存数据到CSV文件
    result_df.to_csv('fsc_data/velocity_time_data.csv', index=False)
    print("Data saved to velocity_time_data.csv")
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(time, velocity, 'b-', linewidth=1)
    
    # 设置图表标题和标签
    plt.title('Vehicle Velocity vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    # 保存图表到文件
    plt.savefig('fsc_data/velocity_time_plot.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    plot_velocity_time()