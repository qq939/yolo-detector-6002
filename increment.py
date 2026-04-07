import os
import cv2
import albumentations as A
import numpy as np
from pathlib import Path

# ================= 增广配置列表 =================
# 增广流水线 (L8)
AUG_PIPELINE = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.GaussianBlur(p=0.1),
]
# 数据集根目录 (L16)
DATASET_ROOT = '/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/vertical shefule.yolov8'
# 每张原图增广的数量 (L18)
AUG_COUNT_PER_IMAGE = 15
# ===============================================

def increment():
    img_dir = Path(DATASET_ROOT) / 'train' / 'images'
    lbl_dir = Path(DATASET_ROOT) / 'train' / 'labels'
    
    # 自动识别是检测(bboxes)还是分割(keypoints/polygons)
    # 这里我们使用 keypoints 来处理多点坐标，因为分割标签的点数是不定的
    # format='xy' 表示原始坐标点
    transform = A.Compose(AUG_PIPELINE, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_aug_' not in f]
    print(f"Found {len(images)} original images. Starting augmentation...")

    for img_name in images:
        img_path = img_dir / img_name
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists(): continue

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取 YOLO 格式的标签
        # 分割格式: class x1 y1 x2 y2 ... (归一化)
        with open(lbl_path, 'r') as f:
            lines = f.read().splitlines()
        
        for i in range(AUG_COUNT_PER_IMAGE):
            all_new_lines = []
            for line in lines:
                parts = list(map(float, line.split()))
                cls = int(parts[0])
                coords = parts[1:]
                
                # 转换归一化坐标到像素坐标
                kpts = []
                for j in range(0, len(coords), 2):
                    kpts.append((coords[j] * w, coords[j+1] * h))
                
                # 执行增广
                augmented = transform(image=image, keypoints=kpts)
                aug_img = augmented['image']
                aug_kpts = augmented['keypoints']
                
                # 转换像素坐标回归一化坐标
                new_coords = []
                for kp in aug_kpts:
                    new_coords.append(kp[0] / w)
                    new_coords.append(kp[1] / h)
                
                all_new_lines.append(f"{cls} {' '.join(map(str, new_coords))}")
            
            # 保存图片 (仅在处理完所有 line 后保存一次图片)
            if 'aug_img' in locals():
                new_img_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                cv2.imwrite(str(img_dir / new_img_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                # 保存标签
                new_lbl_name = f"{img_path.stem}_aug_{i}.txt"
                with open(lbl_dir / new_lbl_name, 'w') as f:
                    f.write('\n'.join(all_new_lines))

    print("Augmentation completed.")

if __name__ == "__main__":
    increment()
