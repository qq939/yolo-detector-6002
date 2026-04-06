import unittest
import subprocess
import time

class TestTrainSeg(unittest.TestCase):
    def test_dry_run(self):
        # 记录开始时间以实现超时机制
        start_time = time.time()
        timeout = 30  # 30秒超时
        
        # 运行 train_seg.py 的干跑模式
        process = subprocess.Popen(['python3', 'train_seg.py', '--dry-run'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.terminate()
                self.fail("train_seg.py 干跑模式运行超时")
            time.sleep(0.5)
            
        stdout, stderr = process.communicate()
        
        # 检查输出中是否包含预期的干跑信息
        self.assertIn("Dry run:", stdout)
        self.assertIn("yolov8n-seg.pt", stdout)
        self.assertIn("segment_dataset/data.yaml", stdout)
        self.assertEqual(process.returncode, 0)

if __name__ == '__main__':
    unittest.main()
