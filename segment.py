import os
import random
import shutil
from pathlib import Path

# Global Parameters
# 数据集根目录 (L10)
DATASET_ROOT = Path('/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/segment_dataset')
# 验证集划分数量 (L12)
VAL_COUNT = 2
# 测试集划分数量 (L14)
TEST_COUNT = 1

def segment_dataset():
    """
    将训练集中的部分样本划分到验证集和测试集
    """
    train_images_dir = DATASET_ROOT / 'train' / 'images'
    train_labels_dir = DATASET_ROOT / 'train' / 'labels'
    
    # 按照 data.yaml 统一使用 valid
    valid_images_dir = DATASET_ROOT / 'valid' / 'images'
    valid_labels_dir = DATASET_ROOT / 'valid' / 'labels'
    test_images_dir = DATASET_ROOT / 'test' / 'images'
    test_labels_dir = DATASET_ROOT / 'test' / 'labels'
    
    # 创建目录
    for d in [valid_images_dir, valid_labels_dir, test_images_dir, test_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 获取所有训练集图片
    images = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < (VAL_COUNT + TEST_COUNT):
        print(f"Error: Not enough images in train ({len(images)}) for splitting.")
        return
        
    random.shuffle(images)
    
    # 划分验证集
    val_samples = images[:VAL_COUNT]
    for img_name in val_samples:
        # 移动图片
        shutil.move(train_images_dir / img_name, valid_images_dir / img_name)
        # 移动对应标签
        label_name = Path(img_name).stem + '.txt'
        if (train_labels_dir / label_name).exists():
            shutil.move(train_labels_dir / label_name, valid_labels_dir / label_name)
        else:
            print(f"Warning: Label not found for {img_name}")
            
    # 划分测试集
    test_samples = images[VAL_COUNT : VAL_COUNT + TEST_COUNT]
    for img_name in test_samples:
        # 移动图片
        shutil.move(train_images_dir / img_name, test_images_dir / img_name)
        # 移动对应标签
        label_name = Path(img_name).stem + '.txt'
        if (train_labels_dir / label_name).exists():
            shutil.move(train_labels_dir / label_name, test_labels_dir / label_name)
        else:
            print(f"Warning: Label not found for {img_name}")
            
    print(f"Successfully segmented: {VAL_COUNT} to val, {TEST_COUNT} to test.")
    print(f"Remaining in train: {len(os.listdir(train_images_dir))}")

if __name__ == '__main__':
    segment_dataset()
