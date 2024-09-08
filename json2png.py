import json
import numpy as np
from PIL import Image, ImageDraw
import os
import glob

def json_to_grayscale(json_file, output_dir):
    # 打印路径以确保它是正确的
    print(f"Opening JSON file: {json_file}")

    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取图像的宽度和高度
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 创建一个空的灰度图像
    grayscale_img = np.zeros((img_height, img_width), dtype=np.uint8)

    # 遍历所有的形状
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        gray_value = label_map.get(label, 0)  # 获取对应的灰度值，默认为0（背景）

        # 确保 points 是一个包含坐标对的列表
        points = [(int(x), int(y)) for x, y in points]

        # 创建一个多边形掩码
        mask = Image.new('L', (img_width, img_height), 0)
        ImageDraw.Draw(mask).polygon(points, outline=gray_value, fill=gray_value)
        mask = np.array(mask)

        # 将掩码应用到灰度图像
        grayscale_img[mask == gray_value] = gray_value

    # 保存灰度图像
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + '.png')
    Image.fromarray(grayscale_img).save(output_file)

def process_all_json_files(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, '*.json'))

    # 对每个 JSON 文件执行操作
    for json_file in json_files:
        json_to_grayscale(json_file, output_dir)

# 定义类别和对应的灰度值
label_map = {
    "background": 0,
    "songsan": 1,
    "liefeng": 2,
    "junlie": 3,
    "kuaixiu": 4,
    "tiaoxiu": 5,
    "aokeng": 6
}
# 示例用法
input_dir = r'./ZH/png2json/'  # 输入目录
output_dir = r'./ZH/json2png/'  # 输出目录
process_all_json_files(input_dir, output_dir)
