import os
import shutil
import unittest
import subprocess
import time

class TestTrainClassSplit(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = 'datasets/train'
        self.cls_dataset_dir = 'yolo_class_dataset'
        
    def test_split_logic(self):
        start_time = time.time()
        timeout = 60
        
        # 运行 train_class.py 的干跑模式
        process = subprocess.Popen(['python3', 'train_class.py', '--dry-run'])
        
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.terminate()
                self.fail("train_class.py 运行超时")
            time.sleep(1)
            
        # 检查分类数据集结构
        # 格式应为: yolo_class_dataset/{train,val}/{ok,ng}/
        for split in ['train', 'val']:
            for cls in ['ok', 'ng']:
                target_dir = os.path.join(self.cls_dataset_dir, split, cls)
                self.assertTrue(os.path.exists(target_dir), f"目录缺失: {target_dir}")
                
        # 检查验证集数量
        val_ok_count = len(os.listdir(os.path.join(self.cls_dataset_dir, 'val/ok')))
        val_ng_count = len(os.listdir(os.path.join(self.cls_dataset_dir, 'val/ng')))
        
        print(f"Validation OK: {val_ok_count}, NG: {val_ng_count}")
        
        self.assertEqual(val_ok_count, 20, f"分类验证集 OK 数量应为 20，实际为 {val_ok_count}")
        self.assertEqual(val_ng_count, 20, f"分类验证集 NG 数量应为 20，实际为 {val_ng_count}")

if __name__ == '__main__':
    unittest.main()
