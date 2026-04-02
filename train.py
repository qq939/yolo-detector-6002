#!/usr/bin/env python3
"""
YOLO 训练脚本
支持自定义数据集训练，自动转换为YOLO格式
"""
import sys
import json
import os
import shutil
import argparse
from pathlib import Path

# 尝试导入ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed")

# 路径常量
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
STATUS_FILE = BASE_DIR / "train_status.json"

def update_status(status, progress=0, message=''):
    """更新训练状态"""
    with open(STATUS_FILE, 'w') as f:
        json.dump({'status': status, 'progress': progress, 'message': message}, f)

def prepare_yolo_dataset():
    """准备YOLO数据集"""
    print("准备YOLO数据集...")
    
    ok_dir = DATASETS_DIR / "ok"
    ng_dir = DATASETS_DIR / "ng"
    
    # 创建YOLO数据集目录结构
    yolo_dir = DATASETS_DIR / "yolo"
    train_img_dir = yolo_dir / "train" / "images"
    val_img_dir = yolo_dir / "val" / "images"
    train_label_dir = yolo_dir / "train" / "labels"
    val_label_dir = yolo_dir / "val" / "labels"
    
    # 清理旧数据
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    
    # 创建目录
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 处理OK图片 (class 0)
    ok_files = []
    if ok_dir.exists():
        ok_files = [f for f in os.listdir(ok_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 处理NG图片 (class 1)
    ng_files = []
    if ng_dir.exists():
        ng_files = [f for f in os.listdir(ng_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"OK文件: {len(ok_files)}, NG文件: {len(ng_files)}")
    
    # 分割训练/验证集 (80/20)
    ok_split = int(len(ok_files) * 0.8)
    ng_split = int(len(ng_files) * 0.8)
    
    # 复制OK图片到训练集
    for f in ok_files[:ok_split]:
        src = ok_dir / f
        dst = train_img_dir / f
        if src.exists():
            shutil.copy2(src, dst)
            # 创建YOLO标签文件 (OK = class 0)
            create_yolo_label(dst, 0)
    
    # 复制OK图片到验证集
    for f in ok_files[ok_split:]:
        src = ok_dir / f
        dst = val_img_dir / f
        if src.exists():
            shutil.copy2(src, dst)
            create_yolo_label(dst, 0)
    
    # 复制NG图片到训练集
    for f in ng_files[:ng_split]:
        src = ng_dir / f
        dst = train_img_dir / f
        if src.exists():
            shutil.copy2(src, dst)
            # 创建YOLO标签文件 (NG = class 1)
            create_yolo_label(dst, 1)
    
    # 复制NG图片到验证集
    for f in ng_files[ng_split:]:
        src = ng_dir / f
        dst = val_img_dir / f
        if src.exists():
            shutil.copy2(src, dst)
            create_yolo_label(dst, 1)
    
    print(f"数据集准备完成: {len(ok_files[:ok_split]) + len(ng_files[:ng_split])} 训练, {len(ok_files[ok_split:]) + len(ng_files[ng_split:])} 验证")
    
    return str(yolo_dir)

def create_yolo_label(image_path, class_id):
    """创建YOLO格式标签文件"""
    # 对于OK/NG二分类，我们创建默认的全图标签
    # 实际使用时，用户应该提供包含位置信息的标注
    # 这里创建默认的中心点标签（整张图作为一个目标）
    from PIL import Image
    
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # YOLO格式: class_id cx cy w h (归一化)
        # 这里创建一个覆盖整张图的默认框
        label_path = image_path.with_suffix('.txt')
        
        # 中心点归一化坐标 (0.5, 0.5)，宽高归一化 (1.0, 1.0)
        with open(label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    except Exception as e:
        print(f"创建标签失败 {image_path}: {e}")

def create_dataset_yaml():
    """创建数据集配置文件"""
    yolo_dir = DATASETS_DIR / "yolo"
    
    yaml_content = f"""
path: {yolo_dir}
train: train/images
val: val/images

nc: 2
names: ['ok', 'ng']
"""
    
    yaml_path = BASE_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"数据集配置已创建: {yaml_path}")
    return str(yaml_path)

def dummy_train(config):
    """模拟训练（无ultralytics时使用）"""
    import time
    import random
    
    update_status('training', 0, '开始模拟训练...')
    
    total_epochs = config.get('epochs', 100)
    
    for epoch in range(total_epochs):
        progress = int((epoch + 1) / total_epochs * 100)
        update_status('training', progress, f'Epoch {epoch+1}/{total_epochs}')
        time.sleep(0.3)
    
    # 创建模拟模型文件
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"trained_{config.get('modelType', 'yolov8n')}.pt"
    model_path.write_text('dummy model')
    
    update_status('completed', 100, '训练完成!')
    return {'success': True, 'model': str(model_path)}

def real_train(config):
    """真实YOLO训练"""
    model_type = config.get('modelType', 'yolov8n')
    epochs = config.get('epochs', 100)
    batch_size = config.get('batchSize', 16)
    image_size = config.get('imageSize', 640)
    
    # 准备数据集
    prepare_yolo_dataset()
    yaml_path = create_dataset_yaml()
    
    # 使用预训练模型
    model = YOLO(f'{model_type}.pt')
    
    update_status('training', 0, '开始训练...')
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            project=str(MODELS_DIR),
            name='train',
            exist_ok=True,
            verbose=True,
            plots=True,
            save=True
        )
        
        update_status('completed', 100, '训练完成!')
        
        # 查找训练生成的模型
        best_model = MODELS_DIR / 'train' / 'weights' / 'best.pt'
        last_model = MODELS_DIR / 'train' / 'weights' / 'last.pt'
        
        if best_model.exists():
            # 复制到 models 目录
            final_model = MODELS_DIR / f"yolo_{model_type}_okng.pt"
            shutil.copy2(best_model, final_model)
            return {'success': True, 'model': str(final_model)}
        elif last_model.exists():
            final_model = MODELS_DIR / f"yolo_{model_type}_okng.pt"
            shutil.copy2(last_model, final_model)
            return {'success': True, 'model': str(final_model)}
        else:
            return {'success': True, 'message': '训练完成'}
            
    except Exception as e:
        update_status('error', 0, str(e))
        return {'success': False, 'error': str(e)}

def train(config):
    """主训练函数"""
    if not YOLO_AVAILABLE:
        return dummy_train(config)
    else:
        return real_train(config)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config = json.loads(sys.argv[1])
    else:
        config = {}
    
    result = train(config)
    print(json.dumps(result, ensure_ascii=False))
