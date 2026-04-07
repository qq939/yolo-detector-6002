import os
import unittest
from pathlib import Path

class TestSplit(unittest.TestCase):
    def setUp(self):
        self.root = Path('/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/vertical shefule.yolov8')
        
    def test_split_ratios(self):
        train_images = list((self.root / 'train/images').glob('*'))
        val_images = list((self.root / 'valid/images').glob('*'))
        test_images = list((self.root / 'test/images').glob('*'))
        
        total = len(train_images) + len(val_images) + len(test_images)
        print(f"Total images: {total}")
        print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # 7:2:1 ratio
        expected_val = int(total * 0.2)
        expected_test = int(total * 0.1)
        expected_train = total - expected_val - expected_test
        
        self.assertEqual(len(val_images), expected_val, f"Validation set should have {expected_val} images, got {len(val_images)}")
        self.assertEqual(len(test_images), expected_test, f"Test set should have {expected_test} images, got {len(test_images)}")
        self.assertEqual(len(train_images), expected_train, f"Training set should have {expected_train} images, got {len(train_images)}")
        
        # 检查标签是否同步
        for split in ['train', 'valid', 'test']:
            img_dir = self.root / split / 'images'
            lbl_dir = self.root / split / 'labels'
            for img_f in img_dir.glob('*'):
                if img_f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    lbl_f = lbl_dir / (img_f.stem + '.txt')
                    self.assertTrue(lbl_f.exists(), f"Label {lbl_f} missing for image {img_f}")

if __name__ == '__main__':
    unittest.main()
