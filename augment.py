import os
import random
import cv2
import albumentations as A
import numpy as np

# Global Parameters
# 训练数据集路径 (L10)
TRAIN_DIR = 'datasets/train'
# 目标样本数量 (L12)
TARGET_COUNT = 50
# 定义增广流程 (L14)
AUG_TRANSFORM = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    A.ISONoise(p=0.1),
    A.RGBShift(p=0.1),
])

def augment_class(class_dir, target_count):
    """
    对指定类别的目录进行样本增广
    """
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"Warning: No images found in {class_dir}")
        return
    
    current_count = len(images)
    print(f"Current count in {class_dir}: {current_count}")
    
    if current_count >= target_count:
        print(f"Count already meets target for {class_dir}")
        return
    
    needed = target_count - current_count
    print(f"Augmenting {needed} images for {class_dir}...")
    
    for i in range(needed):
        # 随机选择一张原图
        img_name = random.choice(images)
        img_path = os.path.join(class_dir, img_name)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # 转换 BGR 到 RGB (albumentations 使用 RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用增广
        augmented = AUG_TRANSFORM(image=image)['image']
        
        # 转换回 BGR 以便保存
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        
        # 生成新文件名
        name, ext = os.path.splitext(img_name)
        new_name = f"{name}_aug_{i}{ext}"
        new_path = os.path.join(class_dir, new_name)
        
        # 保存增广后的图像
        cv2.imwrite(new_path, augmented)

def main():
    # 增广 OK 样本 (L67)
    augment_class(os.path.join(TRAIN_DIR, 'ok'), TARGET_COUNT)
    # 增广 NG 样本 (L69)
    augment_class(os.path.join(TRAIN_DIR, 'ng'), TARGET_COUNT)
    print("Augmentation completed.")

if __name__ == '__main__':
    main()
