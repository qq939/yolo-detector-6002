#!/usr/bin/env python3
"""
YOLO 视频推理脚本
尽可能简单的视频推理实现
"""
import os
import sys
import argparse
from pathlib import Path

# 全局参数定义
# 默认模型 (使用位置: 第 44 行)
DEFAULT_MODEL = "yolov8n.pt"
# 默认输出目录 (使用位置: 第 45 行)
DEFAULT_OUTPUT_DIR = "public/results"
# 置信度阈值 (使用位置: 第 54 行)
CONFIDENCE_THRESHOLD = 0.25

try:
    from ultralytics import YOLO
    import cv2
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics or opencv-python not installed")

def predict_video(video_path, model_path=DEFAULT_MODEL, output_dir=DEFAULT_OUTPUT_DIR):
    """
    对视频进行 YOLO 推理
    """
    if not YOLO_AVAILABLE:
        print("Error: YOLO or OpenCV not available. Please install dependencies.")
        return False
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False

    # 1. 加载模型
    model = YOLO(model_path) # 使用 DEFAULT_MODEL 的逻辑

    # 2. 确保输出目录存在
    output_path = Path(output_dir) # 使用 DEFAULT_OUTPUT_DIR 的逻辑
    output_path.mkdir(parents=True, exist_ok=True)

    # 3. 运行推理
    # YOLO.predict 的流式输出非常简单，它会自动处理视频帧
    # save=True 会将带框的视频保存到 runs/detect/predict 目录
    # 如果要指定输出路径，可以更复杂一点，但这里追求“尽可能简单”
    print(f"Starting inference on {video_path}...")
    results = model.predict(
        source=video_path,
        conf=CONFIDENCE_THRESHOLD, # 使用 CONFIDENCE_THRESHOLD 的逻辑
        save=True,
        project=str(output_path),
        name="video_inference",
        exist_ok=True
    )

    print(f"Inference completed. Results saved in {output_path}/video_inference")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Simple Video Predictor")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Path to model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    success = predict_video(args.video, args.model, args.output)
    if not success:
        sys.exit(1)
    sys.exit(0)
