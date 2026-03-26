import sys
import os
import signal
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

def test_imports():
    try:
        with timeout(30):
            import ultralytics
            from ultralytics import YOLO
            import PIL
            from PIL import Image
            print("Successfully imported ultralytics and PIL")
            return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    except TimeoutException:
        print("Import test timed out")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        sys.exit(0)
    else:
        sys.exit(1)
