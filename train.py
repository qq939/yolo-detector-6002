import os
import random
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# Global Parameters
# 原始数据集路径 (L11)
DATASETS_DIR = Path('datasets/train')
# YOLO格式数据集输出路径 (L13)
YOLO_DATASET_DIR = Path('yolo_dataset')
# 验证集每个类别的抽取数量 (L15)
VAL_SAMPLES_PER_CLASS = 20
# 训练轮数 (L17)
EPOCHS = 50
# 训练使用的模型 (L19)
MODEL_NAME = 'yolov8n.pt'

def prepare_yolo_dataset():
    """
    重新划分并准备 YOLO 格式的数据集
    随机抽取 20 张 OK 和 20 张 NG 到验证集
    """
    print("Preparing YOLO dataset...")
    
    # 清理旧的 yolo_dataset 目录
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)
        
    # 创建目录结构
    for split in ['train', 'val']:
        (YOLO_DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    classes = {'ok': 0, 'ng': 1}
    
    for class_name, class_id in classes.items():
        src_dir = DATASETS_DIR / class_name
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found, skipping...")
            continue
            
        images = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.jpeg')) + list(src_dir.glob('*.png'))
        if len(images) < VAL_SAMPLES_PER_CLASS:
            print(f"Error: {class_name} samples ({len(images)}) less than {VAL_SAMPLES_PER_CLASS}")
            continue
            
        # 随机抽取
        random.shuffle(images)
        val_images = images[:VAL_SAMPLES_PER_CLASS]
        train_images = images[VAL_SAMPLES_PER_CLASS:]
        
        print(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val")
        
        # 处理划分
        for split, split_images in [('train', train_images), ('val', val_images)]:
            for img_path in split_images:
                # 复制图片
                dest_name = f"{class_name}_{img_path.name}"
                dest_img_path = YOLO_DATASET_DIR / 'images' / split / dest_name
                shutil.copy(img_path, dest_img_path)
                
                # 生成标签 (整张图作为检测目标)
                # YOLO 格式: class x_center y_center width height (归一化)
                label_name = f"{class_name}_{img_path.stem}.txt"
                label_path = YOLO_DATASET_DIR / 'labels' / split / label_name
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                    
    # 生成 dataset.yaml
    yaml_content = f"""
path: {YOLO_DATASET_DIR.absolute()}
train: images/train
val: images/val

nc: 2
names:
  0: OK
  1: NG
"""
    with open(YOLO_DATASET_DIR / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
        
    print(f"YOLO dataset prepared at {YOLO_DATASET_DIR}")
    return str(YOLO_DATASET_DIR / 'dataset.yaml')

def main():
    # 检查是否为干跑模式 (仅用于测试划分逻辑) (L86)
    dry_run = '--dry-run' in sys.argv
    
    # 1. 准备数据集 (L89)
    yaml_path = prepare_yolo_dataset()
    
    if dry_run:
        print("Dry run: skipping training.")
        return
        
    # 2. 加载模型 (L95)
    model = YOLO(MODEL_NAME)
    
    # 3. 开始训练 (L98)
    print(f"Starting training for {EPOCHS} epochs...")
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        name='yolo_detector',
        project='runs/train'
    )
    print("Training completed.")

if __name__ == '__main__':
    main()
