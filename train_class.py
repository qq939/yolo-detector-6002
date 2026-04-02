import os
import random
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO

# Global Parameters
# 原始数据集路径 (L10)
DATASETS_DIR = Path('datasets/train')
# 分类数据集输出路径 (L12)
CLS_DATASET_DIR = Path('yolo_class_dataset')
# 验证集每个类别的抽取数量 (L14)
VAL_SAMPLES_PER_CLASS = 20
# 训练轮数 (L16)
EPOCHS = 50
# 训练使用的分类模型 (L18)
MODEL_NAME = 'yolov8n-cls.pt'

def prepare_classification_dataset():
    """
    准备分类模型所需的数据集结构
    root/split/class/image.jpg
    随机抽取 20 张 OK 和 20 张 NG 到验证集
    """
    print("Preparing Classification dataset...")
    
    # 清理旧目录
    if CLS_DATASET_DIR.exists():
        shutil.rmtree(CLS_DATASET_DIR)
        
    classes = ['ok', 'ng']
    for split in ['train', 'val']:
        for cls in classes:
            (CLS_DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)
            
    for cls in classes:
        src_dir = DATASETS_DIR / cls
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found, skipping...")
            continue
            
        images = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.jpeg')) + list(src_dir.glob('*.png'))
        if len(images) < VAL_SAMPLES_PER_CLASS:
            print(f"Error: {cls} samples ({len(images)}) less than {VAL_SAMPLES_PER_CLASS}")
            continue
            
        # 随机抽取
        random.shuffle(images)
        val_images = images[:VAL_SAMPLES_PER_CLASS]
        train_images = images[VAL_SAMPLES_PER_CLASS:]
        
        print(f"Class {cls}: {len(train_images)} train, {len(val_images)} val")
        
        # 复制文件
        for img_path in train_images:
            shutil.copy(img_path, CLS_DATASET_DIR / 'train' / cls / img_path.name)
        for img_path in val_images:
            shutil.copy(img_path, CLS_DATASET_DIR / 'val' / cls / img_path.name)
            
    print(f"Classification dataset prepared at {CLS_DATASET_DIR}")
    return str(CLS_DATASET_DIR.absolute())

def main():
    # 检查是否为干跑模式 (L61)
    dry_run = '--dry-run' in sys.argv
    
    # 1. 准备数据集 (L64)
    dataset_path = prepare_classification_dataset()
    
    if dry_run:
        print("Dry run: skipping training.")
        return
        
    # 2. 加载分类模型 (L70)
    model = YOLO(MODEL_NAME)
    
    # 3. 开始训练 (L73)
    print(f"Starting classification training for {EPOCHS} epochs...")
    model.train(
        data=dataset_path,
        epochs=EPOCHS,
        imgsz=224,
        batch=16,
        name='yolo_classifier',
        project='/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/public/runs/train_cls'
    )
    print("Classification training completed.")

if __name__ == '__main__':
    main()
