import os
import unittest

class TestSegment(unittest.TestCase):
    def setUp(self):
        self.root = 'segment_dataset'
        
    def test_counts(self):
        train_images = os.path.join(self.root, 'train/images')
        val_images = os.path.join(self.root, 'valid/images')
        test_images = os.path.join(self.root, 'test/images')
        
        train_labels = os.path.join(self.root, 'train/labels')
        val_labels = os.path.join(self.root, 'valid/labels')
        test_labels = os.path.join(self.root, 'test/labels')
        
        # 初始 15 张
        # val 2 张
        # test 1 张
        # train 12 张
        
        self.assertEqual(len(os.listdir(val_images)), 2, "Validation images should be 2")
        self.assertEqual(len(os.listdir(test_images)), 1, "Test images should be 1")
        self.assertEqual(len(os.listdir(train_images)), 12, "Train images should be 12")
        
        self.assertEqual(len(os.listdir(val_labels)), 2, "Validation labels should be 2")
        self.assertEqual(len(os.listdir(test_labels)), 1, "Test labels should be 1")
        self.assertEqual(len(os.listdir(train_labels)), 12, "Train labels should be 12")

if __name__ == '__main__':
    unittest.main()
