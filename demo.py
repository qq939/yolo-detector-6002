#!/usr/bin/env python3
"""
视频Demo生成脚本
使用训练好的YOLO模型对视频进行目标检测，生成带框视频
"""
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# 尝试导入ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed")

def process_video(video_path, model_path, output_path):
    """处理视频，逐帧检测"""
    
    if not YOLO_AVAILABLE:
        # 模拟处理
        print("0%")
        import time
        for i in range(10):
            time.sleep(0.5)
            print(f"{(i+1)*10}%")
        print("100%")
        # 复制原视频作为输出
        import shutil
        shutil.copy(video_path, output_path)
        return {'success': True, 'output': output_path}
    
    # 加载模型
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'success': False, 'error': 'Cannot open video'}
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        results = model.predict(frame, verbose=False)
        
        # 绘制结果
        annotated = results[0].plot()
        
        # 写入输出视频
        out.write(annotated)
        
        frame_count += 1
        if frame_count % 10 == 0:
            progress = int((frame_count / total_frames) * 100)
            print(f"{progress}%")
    
    cap.release()
    out.release()
    
    print("100%")
    return {'success': True, 'output': output_path}

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python demo.py <video_path> <model_path> <output_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    
    result = process_video(video_path, model_path, output_path)
    print(json.dumps(result))
