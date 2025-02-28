from gen_spectrum import Bartlett
from data_painter import paint_spectrum, paint_spectrum_linear_array, paint_pattern_spectrum_linear_array
from PIL import Image
import numpy as np
import imageio.v2 as imageio
import torch


if __name__ == '__main__':
    
    # #先用巴特莱特算法产生黑白的空间谱
    # sample_phase = [-1.2, -0.8, 0.3, 1.5]  # 替换为你的实际测量值
    # worker = Bartlett()
    # spectrum = worker.gen_spectrum_linear(sample_phase)
    # imageio.imsave('dataset_tools/imgs/linear_spectrum.png', spectrum)

    # #再读取图片变为极坐标的
    # save_path = "dataset_tools/imgs/linear_spectrum_ploar_grid.png"
    # spectrum = Image.open("dataset_tools/imgs/linear_spectrum.png")
    # spectrum = np.array(spectrum)
    # spectrum = torch.tensor(spectrum)
    # paint_spectrum_linear_array(spectrum, save_path)


    #一次性生成极坐标形式的空间谱
    save_path_1 = "dataset_tools/imgs/linear_spectrum_ploar_grid-2.png"
    save_path_2 = "dataset_tools/imgs/linear_spectrum_ploar_grid-3.png"

    sample_phase = [-1.34, -0.865, 0.324, 0.57]  # 替换为你的实际测量值
    worker = Bartlett()
    spectrum = worker.gen_spectrum_linear(sample_phase)
    spectrum = torch.tensor(spectrum)
    # paint_spectrum_linear_array(spectrum, save_path_1)
    main_lobe_angle = paint_pattern_spectrum_linear_array(spectrum, save_path_2)
    print(f"主瓣角度: {main_lobe_angle}°")