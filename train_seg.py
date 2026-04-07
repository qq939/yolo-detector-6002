import sys
from ultralytics import YOLO

# Global Parameters
# 数据集配置文件路径 (L7)
DATA_YAML = 'vertical shefule.yolov8/data.yaml'
# 训练轮数 (L9)
EPOCHS = 50
# 训练使用的分割模型 (L11)
MODEL_NAME = 'yolov8n-seg.pt'
# 训练结果保存路径 (L13)
PROJECT_PATH = '/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/public/runs/train_seg_2'

def main():
    # 检查是否为干跑模式 (L17)
    dry_run = '--dry-run' in sys.argv
    
    if dry_run:
        print(f"Dry run: Using model {MODEL_NAME} with data {DATA_YAML}")
        return
        
    # 1. 加载分割模型 (L23)
    model = YOLO(MODEL_NAME)
    
    # 2. 开始训练 (L26)
    print(f"Starting segmentation training for {EPOCHS} epochs...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        name='yolo_segmenter',
        project=PROJECT_PATH
    )
    print("Segmentation training completed.")

if __name__ == '__main__':
    main()
