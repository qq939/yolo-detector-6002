import os
import unittest
import subprocess
import time

class TestIncrement(unittest.TestCase):
    def test_increment_logic(self):
        # 记录开始时间以实现超时机制
        start_time = time.time()
        timeout = 60  # 60秒超时
        
        # 运行增广脚本 (干跑模式或直接运行)
        # 注意：increment.py 应该对 train/images 进行增广
        process = subprocess.Popen(['python3', 'increment.py'])
        
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.terminate()
                self.fail("increment.py 运行超时")
            time.sleep(1)
            
        # 检查运行结果
        img_dir = '/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/vertical shefule.yolov8/train/images'
        label_dir = '/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/vertical shefule.yolov8/train/labels'
        
        files = os.listdir(img_dir)
        aug_files = [f for f in files if '_aug_' in f]
        
        print(f"Total images: {len(files)}, Augmented: {len(aug_files)}")
        
        self.assertGreater(len(aug_files), 0, "没有生成增广样本")
        
        # 验证对应的标签文件是否存在
        for img_f in aug_files:
            label_f = os.path.splitext(img_f)[0] + '.txt'
            self.assertTrue(os.path.exists(os.path.join(label_dir, label_f)), f"标签文件缺失: {label_f}")

if __name__ == '__main__':
    unittest.main()
