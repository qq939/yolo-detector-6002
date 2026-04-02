import os
import shutil
import unittest
import subprocess
import time

class TestTrainSplit(unittest.TestCase):
    def setUp(self):
        # 确保数据集存在
        self.dataset_dir = 'datasets'
        self.yolo_dir = 'yolo_dataset'
        
    def test_split_logic(self):
        # 记录开始时间以实现超时机制
        start_time = time.time()
        timeout = 120  # 120秒超时，因为可能包含训练初始化的部分
        
        # 我们只需要运行 train.py 的划分逻辑部分
        # 为了测试方便，我们可以修改 train.py 支持 --split-only 参数
        # 或者直接运行并检查 yolo_dataset 目录
        
        # 运行 train.py
        # 注意：实际训练可能很慢，我们希望 train.py 在划分完数据后能被我们检测到
        process = subprocess.Popen(['python3', 'train.py', '--dry-run'])
        
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.terminate()
                self.fail("train.py 运行超时")
            time.sleep(1)
            
        # 检查验证集数量
        val_ok_dir = os.path.join(self.yolo_dir, 'images/val')
        val_labels_dir = os.path.join(self.yolo_dir, 'labels/val')
        
        ok_val_files = [f for f in os.listdir(val_ok_dir) if f.startswith('ok_')]
        ng_val_files = [f for f in os.listdir(val_ok_dir) if f.startswith('ng_')]
        
        print(f"Validation OK: {len(ok_val_files)}, NG: {len(ng_val_files)}")
        
        self.assertEqual(len(ok_val_files), 20, f"验证集 OK 数量应为 20，实际为 {len(ok_val_files)}")
        self.assertEqual(len(ng_val_files), 20, f"验证集 NG 数量应为 20，实际为 {len(ng_val_files)}")
        
        # 检查标签文件是否存在
        for img_f in ok_val_files:
            label_f = os.path.splitext(img_f)[0] + '.txt'
            self.assertTrue(os.path.exists(os.path.join(val_labels_dir, label_f)), f"标签文件缺失: {label_f}")

if __name__ == '__main__':
    unittest.main()
