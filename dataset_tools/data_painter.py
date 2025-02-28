# -*- coding: utf-8 -*-
"""painter for data
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def paint_spectrum(spectrum, save_path=None):

    spectrum = spectrum.numpy().reshape(90, 360)
    plt.imsave(save_path, spectrum, cmap='jet')
    spectrum = np.flipud(spectrum)
    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.*np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    cax = ax.pcolormesh(theta, r, spectrum.T, cmap='jet', shading='flat')
    ax.axis('off')

    # save the image as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

def paint_spectrum_linear_array(spectrum, save_path=None):
    """
    将线性阵列的空间谱转换为极坐标形式并保存图像
    :param spectrum: 输入的空间谱（181x100）
    :param save_path: 图像保存路径
    """
    # 将频谱转换为numpy数组并reshape为181x100
    spectrum = spectrum.numpy().reshape(181, 100)
    plt.imsave(save_path, spectrum, cmap='jet')
    # 将频谱上下翻转（符合极坐标显示习惯）
    spectrum = np.flipud(spectrum)
    
    # 创建极坐标网格
    r = np.linspace(0, 1, 101)  # 径向距离（0到1，对应100行）
    theta = np.linspace(0, np.pi, 182)  # 角度范围（0到π，对应181列）
    
    # 生成网格
    r, theta = np.meshgrid(r, theta)
    
    # 创建极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 绘制半圆极坐标图
    cax = ax.pcolormesh(theta, r, spectrum, cmap='jet', shading='flat')
    # 隐藏坐标轴
    ax.axis('off')
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

def paint_pattern_spectrum_linear_array(spectrum, save_path=None):
    """
    将线性阵列的空间谱转换为极坐标形式并保存图像，用半径表示信号强度
    :param spectrum: 输入的空间谱（181x100）
    :param save_path: 图像保存路径
    """
    # 将频谱转换为numpy数组并reshape为181x100
    spectrum = spectrum.numpy().reshape(1810, 100)
    
    # 将频谱上下翻转（符合极坐标显示习惯）
    spectrum = np.flipud(spectrum)
    
    # 创建极坐标网格
    theta = np.linspace(0, np.pi, 1810)  # 角度范围（0到π，对应181列）
    r = np.linspace(0, 1, 100)         # 径向距离（0到1，对应100行）
    
    # 生成网格
    theta, r = np.meshgrid(theta, r)  # theta 在前，r 在后，生成 (100, 181) 的网格
    
    # 转置 spectrum，使其维度与 theta 和 r 匹配
    spectrum = spectrum / np.max(spectrum) * 100 #值归一化到0～100
    spectrum = spectrum.T  # 转置后维度为 (100, 181)

    # 找到主瓣角度
    max_value = np.max(spectrum)  # 找到 spectrum 中的最大值
    max_index = np.unravel_index(np.argmax(spectrum), spectrum.shape)  # 找到最大值的位置
    main_lobe_angle_rad = theta[max_index]  # 主瓣角度（弧度）
    main_lobe_angle_deg = np.rad2deg(main_lobe_angle_rad)  # 主瓣角度（度）
    
    # 创建极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 绘制半圆形背景
    draw_semi_circle_grid(ax)
    
    # 绘制极坐标图，用半径表示信号强度
    for i in range(spectrum.shape[0]):  # 遍历每个径向距离
        ax.plot(theta[i, :], spectrum[i, :], color='blue', linewidth=1)  # 用蓝色线条绘制
    
    # 设置极坐标图的角度范围为 [0, π]
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    return main_lobe_angle_deg

def draw_semi_circle_grid(ax):
    """
    绘制半圆形背景，包含同心圆弧和半径构成的扇形格子，半径范围为 0 到 90
    :param ax: matplotlib 的 Axes 对象
    """
    # 绘制同心圆弧
    radii = np.linspace(0, 100, 5)  # 同心圆弧的半径
    for r in radii:
        ax.plot(np.linspace(0, np.pi, 100), np.ones(100) * r, color='black', linewidth=1, alpha=0.5)
    
    # 绘制半径
    angles = np.linspace(0, np.pi, 7)  # 半径的角度（0到π，共7个）
    for angle in angles:
        ax.plot([angle, angle], [0, 100], color='black', linewidth=1, alpha=0.5)
    
    # 添加角度标记
    for angle in [-90, -60, -30, 0, 30, 60, 90]:
        ax.text(np.deg2rad(angle), 95, f"{angle}°", ha='center', va='center', fontsize=3, alpha=0.7)
    


def paint_spectrum_compare(pred_spectrum, gt_spectrum, save_path=None):

    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.*np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    cax1 = axs[0].pcolormesh(theta, r, np.flipud(pred_spectrum).T, cmap='viridis', shading='flat')
    axs[0].axis('off')

    cax2 = axs[1].pcolormesh(theta, r, np.flipud(gt_spectrum).T, cmap='viridis', shading='flat')
    axs[1].axis('off')

    # save the image as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def paint_location(loc_path, save_path):


    all_loc = np.loadtxt(os.path.join(loc_path, 'tx_pos.csv'), delimiter=',', skiprows=1)
    train_index = np.loadtxt(os.path.join(loc_path, 'train_index.txt'), dtype=int)
    test_index = np.loadtxt(os.path.join(loc_path, 'test_index.txt'), dtype=int)
    train_loc = all_loc[train_index-1]
    test_loc = all_loc[test_index-1]
    plt.scatter(train_loc[:, 0], train_loc[:, 1], c='b', label='train',s=0.1)
    plt.scatter(test_loc[:, 0], test_loc[:, 1], c='r', label='test',s=0.1)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loc.pdf'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    #用于画出数据集中的train/test中tx_pos的位置分布的
    loc_path = "data/RFID/s23/"
    save_path = "data/RFID/s23/"
    paint_location(loc_path, save_path)