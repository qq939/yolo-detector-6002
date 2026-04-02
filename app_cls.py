import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Global Parameters
# 分类模型权重路径 (L11)
MODEL_PATH = '/Users/jimjiang/.openclaw/workspace/yolo-detector-6002/public/runs/train_cls/yolo_classifier/weights/best.pt'
# 上传目录 (L13)
UPLOAD_FOLDER = 'public/uploads'
# 允许的文件扩展名 (L15)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
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
        
        # 获取结果 (分类模型)
        result = results[0]
        probs = result.probs
        top1_idx = probs.top1
        top1_conf = float(probs.top1conf)
        top1_label = result.names[top1_idx]
        
        return jsonify({
            'success': True,
            'label': top1_label,
            'confidence': top1_conf,
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return send_from_directory('public', 'index_cls.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
