import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from pathlib import Path

# Global Parameters
# 分割模型权重路径 (L12)
MODEL_PATH = '/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/public/runs/train_seg_2/yolo_segmenter/weights/last.pt'
# 上传目录 (L14)
UPLOAD_FOLDER = 'public/uploads'
# 允许的文件扩展名 (L16)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型
try:
    model = YOLO(MODEL_PATH)
    print(f"Segmentation model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading segmentation model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict_seg', methods=['POST'])
def predict_seg():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 执行推理
        results = model.predict(filepath)
        
        # 保存结果图
        result = results[0]
        print("-"*1000)
        print(result)
        print("-"*1000)
        res_filename = f"res_{filename}"
        res_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)
        
        # plot() 会返回带有 mask 和 label 的图像
        res_img = result.plot(boxes=False, masks=True, labels=True)
        cv2.imwrite(res_path, res_img)
        
        # 获取检测到的类别
        detected_names = []
        if result.boxes:
            for c in result.boxes.cls:
                detected_names.append(result.names[int(c)])
        
        return jsonify({
            'success': True,
            'original_image': filename,
            'result_image': res_filename,
            'detected': detected_names
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return send_from_directory('public', 'index_seg.html')

if __name__ == '__main__':
    # 使用 5002 端口
    app.run(host='0.0.0.0', port=5002, debug=True)
