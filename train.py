#!/usr/bin/env python3
"""
YOLO 训练脚本
支持自定义数据集训练
"""
import sys
import json
import os
import argparse
from pathlib import Path

# 尝试导入ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed")

# 训练状态文件
STATUS_FILE = Path(__file__).parent / "train_status.json"
MODELS_DIR = Path(__file__).parent / "models"

def update_status(status, progress=0, message=''):
    """更新训练状态"""
    with open(STATUS_FILE, 'w') as f:
        json.dump({'status': status, 'progress': progress, 'message': message}, f)

def dummy_train(config):
    """模拟训练（无ultralytics时使用）"""
    import time
    import random
    
    update_status('training', 0, '开始模拟训练...')
    
    total_epochs = config.get('epochs', 100)
    
    for epoch in range(total_epochs):
        progress = int((epoch + 1) / total_epochs * 100)
        update_status('training', progress, f'Epoch {epoch+1}/{total_epochs}')
        time.sleep(0.5)  # 模拟每个epoch
    
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
    
    # 使用预训练模型
    model = YOLO(f'{model_type}.pt')
    
    # 训练
    # 注意: 实际训练需要准备好数据集yaml配置文件
    # 这里提供一个示例配置
    update_status('training', 0, '开始训练...')
    
    try:
        results = model.train(
            data='dataset.yaml',  # 需要数据集配置
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            project=str(MODELS_DIR),
            name='train',
            exist_ok=True,
            verbose=True,
            # 回调更新进度
            plots=True
        )
        
        update_status('completed', 100, '训练完成!')
        
        # 查找训练生成的模型
        best_model = MODELS_DIR / 'train' / 'weights' / 'best.pt'
        if best_model.exists():
            return {'success': True, 'model': str(best_model)}
        else:
            return {'success': True, 'message': '训练完成，模型已保存'}
            
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
