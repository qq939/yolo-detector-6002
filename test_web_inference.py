import os
import requests
import unittest
import subprocess
import time

class TestWebInference(unittest.TestCase):
    def test_predict_api(self):
        # 记录开始时间以实现超时机制
        start_time = time.time()
        timeout = 60  # 60秒超时
        
        # 启动 Flask 应用
        process = subprocess.Popen(['python3', 'app_cls.py'])
        
        # 等待应用启动
        max_retries = 10
        app_started = False
        for i in range(max_retries):
            try:
                response = requests.get('http://localhost:5001/')
                if response.status_code == 200:
                    app_started = True
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
            
        if not app_started:
            process.terminate()
            self.fail("Flask 应用未在预期时间内启动")
            
        # 准备一张测试图片
        test_img_dir = 'datasets/train/ok'
        test_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not test_images:
            process.terminate()
            self.fail("未找到测试图片")
            
        test_img_path = os.path.join(test_img_dir, test_images[0])
        
        # 发送预测请求
        with open(test_img_path, 'rb') as f:
            files = {'file': (test_images[0], f, 'image/jpeg')}
            response = requests.post('http://localhost:5001/predict', files=files)
            
        # 停止 Flask 应用
        process.terminate()
        
        # 检查响应
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('label', data)
        self.assertIn('confidence', data)
        print(f"API Response: {data}")

if __name__ == '__main__':
    unittest.main()
