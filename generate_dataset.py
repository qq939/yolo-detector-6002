#!/usr/bin/env python3
"""
生成YOLO格式数据集
从OK/NG图片生成训练用的数据集
"""
import sys
import json
import os
import shutil
import random
from pathlib import Path
from PIL import Image

# YOLO类别: 0=OK, 1=NG
CLASSES = ['OK', 'NG']

def create_yolo_dataset(config):
    """创建YOLO格式数据集"""
    dataset_dir = Path(__file__).parent / 'datasets'
    output_dir = Path(__file__).parent / 'yolo_dataset'
    
    # 创建目录
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 处理OK图片 (class 0)
    ok_dir = dataset_dir / 'ok'
    if ok_dir.exists():
        ok_images = list(ok_dir.glob('*.jpg')) + list(ok_dir.glob('*.jpeg')) + list(ok_dir.glob('*.png'))
        print(f"Found {len(ok_images)} OK images")
        
        # 分割训练集和验证集
        random.shuffle(ok_images)
        train_count = int(len(ok_images) * 0.8)
        
        for i, img_path in enumerate(ok_images):
            is_train = i < train_count
            split = 'train' if is_train else 'val'
            
            # 复制图片
            dest_img = output_dir / 'images' / split / f'ok_{i}{img_path.suffix}'
            shutil.copy(img_path, dest_img)
            
            # 生成标签 (整张图都是OK)
            width, height = Image.open(img_path).size
            label_path = output_dir / 'labels' / split / f'ok_{i}.txt'
            # YOLO格式: class x_center y_center width height (归一化)
            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 1.0 1.0\n")
    
    # 处理NG图片 (class 1)
    ng_dir = dataset_dir / 'ng'
    if ng_dir.exists():
        ng_images = list(ng_dir.glob('*.jpg')) + list(ng_dir.glob('*.jpeg')) + list(ng_dir.glob('*.png'))
        print(f"Found {len(ng_images)} NG images")
        
        random.shuffle(ng_images)
        train_count = int(len(ng_images) * 0.8)
        
        for i, img_path in enumerate(ng_images):
            is_train = i < train_count
            split = 'train' if is_train else 'val'
            
            # 复制图片
            dest_img = output_dir / 'images' / split / f'ng_{i}{img_path.suffix}'
            shutil.copy(img_path, dest_img)
            
            # 生成标签 (整张图都是NG)
            width, height = Image.open(img_path).size
            label_path = output_dir / 'labels' / split / f'ng_{i}.txt'
            with open(label_path, 'w') as f:
                f.write(f"1 0.5 0.5 1.0 1.0\n")
    
    # 生成dataset.yaml
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

nc: 2
names:
  0: OK
  1: NG
"""
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset created at {output_dir}")
    return str(output_dir)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config = json.loads(sys.argv[1])
    else:
        config = {}
    
    result = create_yolo_dataset(config)
    print(json.dumps({'success': True, 'dataset': result}))
