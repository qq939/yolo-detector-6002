#!/usr/bin/env python3
"""
YOLO 推理脚本
支持图片和批量推理，输出带检测框的结果图片
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
    print("Warning: ultralytics not installed, using dummy mode")

# 结果输出目录
RESULTS_DIR = Path(__file__).parent / "public" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def dummy_predict(image_path):
    """模拟推理（无ultralytics时使用）"""
    import random
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.open(image_path)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # 随机生成一些检测结果
    classes = ['缺陷', '正常', '异常', '划痕', '污点']
    detections = []
    
    num_detections = random.randint(0, 3)
    for _ in range(num_detections):
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        w = random.randint(50, 200)
        h = random.randint(50, 200)
        cls = random.choice(classes)
        conf = random.uniform(0.5, 0.98)
        
        # 画框
        draw.rectangle([x1, y1, x1 + w, y1 + h], outline='red', width=3)
        draw.text((x1, y1 - 20), f'{cls} {conf:.2f}', fill='red')
        
        detections.append({
            'class': cls,
            'confidence': conf,
            'bbox': [x1, y1, x1 + w, y1 + h]
        })
    
    # 保存结果
    output_name = f"result_{Path(image_path).stem}.jpg"
    output_path = RESULTS_DIR / output_name
    img.save(output_path)
    
    return {
        'image': str(output_path),
        'detections': detections,
        'total': len(detections)
    }

def real_predict(image_path, model_path=None):
    """真实YOLO推理"""
    if model_path and os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        # 使用默认预训练模型
        model = YOLO('yolov8n.pt')
    
    # 推理
    results = model.predict(image_path, save=False, save_txt=False)
    
    result = results[0]
    
    # 提取检测结果
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            detections.append({
                'class': result.names[int(boxes.cls[i])],
                'confidence': float(boxes.conf[i]),
                'bbox': boxes.xyxy[i].tolist()
            })
    
    # 保存带框图片
    output_name = f"result_{Path(image_path).stem}.jpg"
    output_path = RESULTS_DIR / output_name
    
    # 绘制结果
    annotated = result.plot()
    Image.fromarray(annotated).save(output_path)
    
    return {
        'image': str(output_path),
        'detections': detections,
        'total': len(detections)
    }

def batch_predict(image_paths):
    """批量推理"""
    results = []
    for path in image_paths:
        try:
            result = predict(path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            results.append({'image': path, 'error': str(e), 'detections': []})
    
    return results

def predict(image_path, model_path=None):
    """主预测函数"""
    if not YOLO_AVAILABLE:
        return dummy_predict(image_path)
    else:
        return real_predict(image_path, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Image path or comma-separated paths for batch')
    parser.add_argument('--batch', action='store_true', help='Batch mode')
    parser.add_argument('--model', help='Model path')
    args = parser.parse_args()
    
    if args.batch:
        # 批量模式
        images = args.image.split(',')
        results = batch_predict(images)
        print(json.dumps(results, ensure_ascii=False))
    else:
        # 单张模式
        result = predict(args.image, args.model)
        print(json.dumps(result, ensure_ascii=False))
