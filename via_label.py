import os
import numpy as np
from PIL import Image

# 定义类别数
num_classes = 7

# 创建一个颜色映射表
color_map = np.array([
    [0, 0, 0],       # 背景 Background
    [255, 0, 0],     # 类别1 Class 1  红 songsan
    [0, 255, 0],     # 类别2 Class 2  绿 liefeng
    [0, 0, 255],     # 类别3 Class 3  蓝 junlie
    [255, 255, 0],   # 类别4 Class 4  黄 kuaixiu
    [0, 255, 255],   # 类别5 Class 5  青 tiaoxiu
    [255, 0, 255]    # 类别6 Class 6  洋红 aokeng
    # 类别7 Class 7 可以选择其他颜色，比如 [128, 128, 128]
])

# 文件夹路径
folder_path = r"./out_t"
out_folder_path = r'./'

# 遍历文件夹中的所有图片
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        label_image_path = os.path.join(folder_path, filename)
        
        # 加载标签图像
        label_image = Image.open(label_image_path)
        
        # 将标签图像转换为numpy数组
        label_array = np.array(label_image)
        
        # 确保标签图像的shape是(H,W)
        assert label_array.ndim == 2
        
        # 使用颜色映射将标签图像转换为RGB图像
        vis_image = color_map[label_array]
        
        # 将numpy数组转换回PIL图像对象
        vis_image = Image.fromarray(vis_image.astype('uint8'))
        
        # 保存可视化后的图像
        vis_image.save(os.path.join(out_folder_path, filename))
        print(f"Processed {filename}")
