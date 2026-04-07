import os
import random
import shutil
from pathlib import Path

# ================= 划分配置 =================
# 数据集根目录 (L7)
DATASET_ROOT = Path('/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/vertical shefule.yolov8')
# 划分比例 7:2:1 (L9)
RATIO = [0.7, 0.2, 0.1]
# ===========================================

def split_dataset():
    """
    按照 7:2:1 的比例重新划分数据集，余数归入训练集
    """
    # 汇总所有图片和标签 (由于目前都在 train 目录下)
    all_img_dir = DATASET_ROOT / 'train' / 'images'
    all_lbl_dir = DATASET_ROOT / 'train' / 'labels'
    
    # 确保目标目录存在
    for split in ['train', 'valid', 'test']:
        (DATASET_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    images = [f for f in os.listdir(all_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(images)
    print(f"Total images found: {total}")
    
    if total == 0:
        print("No images found to split.")
        return
    
    random.shuffle(images)
    
    # 计算各部分数量
    val_count = int(total * RATIO[1])
    test_count = int(total * RATIO[2])
    train_count = total - val_count - test_count
    
    print(f"Splitting: Train={train_count}, Valid={val_count}, Test={test_count}")
    
    # 划分列表
    splits = {
        'valid': images[:val_count],
        'test': images[val_count : val_count + test_count],
        'train': images[val_count + test_count :]
    }
    
    # 执行移动 (如果已经在目标位置则跳过，否则移动)
    for split, split_images in splits.items():
        dest_img_dir = DATASET_ROOT / split / 'images'
        dest_lbl_dir = DATASET_ROOT / split / 'labels'
        
        for img_name in split_images:
            src_img_path = all_img_dir / img_name
            src_lbl_path = all_lbl_dir / (Path(img_name).stem + '.txt')
            
            # 只有当源文件存在且不在目标位置时才移动
            if src_img_path.exists() and src_img_path != dest_img_dir / img_name:
                shutil.move(src_img_path, dest_img_dir / img_name)
            
            if src_lbl_path.exists() and src_lbl_path != dest_lbl_dir / src_lbl_path.name:
                shutil.move(src_lbl_path, dest_lbl_dir / src_lbl_path.name)
                
    # 更新 data.yaml (使用绝对路径以确保稳定)
    data_yaml_path = DATASET_ROOT / 'data.yaml'
    yaml_content = f"""
path: {DATASET_ROOT.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['Black metal plate', 'shiney metal plate', 'spacer']
"""
    with open(data_yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print("Split completed and data.yaml updated.")

if __name__ == "__main__":
    split_dataset()
