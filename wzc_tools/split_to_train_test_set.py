import json
import numpy as np

def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r') as f:
        return json.load(f)

def split_into_scenes(data, scene_length=25, num_scenes=55):
    """将数据划分为场景"""
    # 只取需要的数据长度
    required_length = scene_length * num_scenes
    data = data[:required_length]
    
    # 将数据reshape成(num_scenes, scene_length)的形式
    scenes = []
    for i in range(num_scenes):
        start_idx = i * scene_length
        end_idx = start_idx + scene_length
        scenes.append(data[start_idx:end_idx])
    
    return scenes

def create_train_test_sets(scenes, train_samples=20, test_samples=5):
    """从每个场景中创建训练集和测试集，同时返回索引"""
    train_set = []
    test_set = []
    train_indices = []
    test_indices = []
    
    for scene_idx, scene in enumerate(scenes):
        # 计算场景的基础索引
        base_idx = scene_idx * 25
        
        # 训练集
        train_set.extend(scene[:train_samples])
        train_indices.extend([base_idx + i for i in range(train_samples)])
        
        # 测试集
        test_set.extend(scene[train_samples:train_samples + test_samples])
        test_indices.extend([base_idx + i for i in range(train_samples, train_samples + test_samples)])
    
    return train_set, test_set, train_indices, test_indices

def save_datasets(train_set, test_set, train_path, test_path):
    """保存训练集和测试集到JSON文件"""
    with open(train_path, 'w') as f:
        json.dump(train_set, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_set, f, indent=2)

def save_indices(indices, file_path):
    """保存索引到文本文件"""
    with open(file_path, 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")

def main():
    # 文件路径
    input_file = 'fsc_data/combined_gnss_data.json'
    train_output = 'fsc_data/train_set.json'
    test_output = 'fsc_data/test_set.json'
    train_indices_output = 'fsc_data/train_index.txt'
    test_indices_output = 'fsc_data/test_index.txt'
    
    # 加载数据
    data = load_data(input_file)
    
    # 划分场景
    scenes = split_into_scenes(data)
    
    # 创建训练集和测试集，同时获取索引
    train_set, test_set, train_indices, test_indices = create_train_test_sets(scenes)
    
    # 保存数据集
    save_datasets(train_set, test_set, train_output, test_output)
    
    # 保存索引
    save_indices(train_indices, train_indices_output)
    save_indices(test_indices, test_indices_output)
    
    # 打印数据集信息
    print(f"Total scenes: {len(scenes)}")
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Indices saved to {train_indices_output} and {test_indices_output}")

if __name__ == "__main__":
    main()
