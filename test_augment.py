import os
import subprocess
import time
import unittest

class TestAugment(unittest.TestCase):
    def test_augmentation_count(self):
        # 记录开始时间以实现超时机制
        start_time = time.time()
        timeout = 60  # 60秒超时
        
        # 运行增广脚本
        process = subprocess.Popen(['python3', 'augment.py'])
        
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.terminate()
                self.fail("增广脚本运行超时")
            time.sleep(1)
            
        # 检查运行结果
        ok_dir = 'datasets/train/ok'
        ng_dir = 'datasets/train/ng'
        
        ok_count = len([f for f in os.listdir(ok_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        ng_count = len([f for f in os.listdir(ng_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"OK samples: {ok_count}, NG samples: {ng_count}")
        
        self.assertGreaterEqual(ok_count, 50, f"OK 样本数量不足: {ok_count}")
        self.assertGreaterEqual(ng_count, 50, f"NG 样本数量不足: {ng_count}")

if __name__ == '__main__':
    unittest.main()
