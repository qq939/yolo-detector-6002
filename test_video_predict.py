import sys
import os
import signal
import subprocess
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Test timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def test_video_predict_help():
    """测试脚本是否能正常运行并显示帮助信息"""
    try:
        with timeout(10):
            result = subprocess.run(
                [sys.executable, "video_predict.py", "--help"],
                capture_output=True,
                text=True,
                check=True
            )
            print("Successfully ran video_predict.py --help")
            return "usage" in result.stdout.lower()
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_video_predict_import():
    """测试脚本的导入是否正常"""
    try:
        with timeout(30):
            # 仅仅检查脚本是否可以被 python 解析（即 import 没问题）
            result = subprocess.run(
                [sys.executable, "video_predict.py", "--help"],
                capture_output=True,
                text=True,
                check=True
            )
            print("Import/Execution check passed")
            return True
    except Exception as e:
        print(f"Import/Execution check failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    if not test_video_predict_help():
        print("Help test failed")
        success = False
    if not test_video_predict_import():
        print("Import test failed")
        success = False
    
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        sys.exit(1)
