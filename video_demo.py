#!/usr/bin/env python3
"""
YOLO 视频Demo生成脚本
逐帧检测并生成带检测框的输出视频
"""
import sys
import os
import json
from pathlib import Path
import cv2

# 尝试导入ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed")

# 路径
RESULTS_DIR = Path(__file__).parent / "public" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def get_model(model_path):
    """获取模型"""
    if YOLO_AVAILABLE:
        if model_path and os.path.exists(model_path):
            return YOLO(model_path)
        else:
            # 尝试使用训练好的模型
            models_dir = Path(__file__).parent / "models"
            if models_dir.exists():
                models = list(models_dir.glob("*.pt"))
                if models:
                    return YOLO(str(models[0]))
            # 默认使用预训练模型
            return YOLO('yolov8n.pt')
    return None

def process_video(video_path, model_path, job_id):
    """处理视频"""
    print(f"处理视频: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    # 获取模型
    model = get_model(model_path)
    
    # 创建输出视频
    output_name = f"demo_{Path(video_path).stem}_{job_id}.mp4"
    output_path = RESULTS_DIR / output_name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    detections_log = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 每隔几帧处理一次，提高速度
        if frame_count % 3 == 0:
            if YOLO_AVAILABLE and model:
                # 真实推理
                results = model.predict(frame, save=False, verbose=False)
                result = results[0]
                
                # 绘制检测框
                annotated = result.plot()
                out.write(annotated)
                
                # 记录检测结果
                if result.boxes is not None:
                    for box in result.boxes:
                        detections_log.append({
                            'frame': frame_count,
                            'class': result.names[int(box.cls[0])],
                            'conf': float(box.conf[0])
                        })
            else:
                # 模拟推理
                out.write(frame)
        else:
            out.write(frame)
        
        # 打印进度
        if frame_count % 30 == 0:
            progress = int(frame_count / total_frames * 100)
            print(f"进度: {progress}% ({frame_count}/{total_frames})")
    
    cap.release()
    out.release()
    
    # 保存检测日志
    log_path = RESULTS_DIR / f"detections_{job_id}.json"
    with open(log_path, 'w') as f:
        json.dump(detections_log, f, ensure_ascii=False)
    
    print(f"视频处理完成: {output_path}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python video_demo.py <video_path> <model_path> [job_id]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    job_id = sys.argv[3] if len(sys.argv) > 3 else 'unknown'
    
    success = process_video(video_path, model_path, job_id)
    sys.exit(0 if success else 1)
