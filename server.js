const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 6002;

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use('/downloads', express.static(path.join(__dirname, 'public/results')));

// 确保目录存在
const dirs = ['public/uploads', 'public/results', 'datasets', 'models'];
dirs.forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Multer配置 - 上传图片
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'public/uploads');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

// Multer配置 - 上传数据集
const datasetStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'datasets');
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  }
});
const uploadDataset = multer({ storage: datasetStorage });

// ============ 训练相关API ============

// 开始训练
app.post('/api/train', uploadDataset.single('dataset'), (req, res) => {
  const { epochs, batchSize, imageSize, modelType } = req.body;
  
  const config = {
    epochs: parseInt(epochs) || 100,
    batchSize: parseInt(batchSize) || 16,
    imageSize: parseInt(imageSize) || 640,
    modelType: modelType || 'yolov8n'
  };

  console.log('开始训练，配置:', config);

  // 启动训练
  const trainScript = spawn('python3', ['train.py', JSON.stringify(config)], {
    cwd: __dirname,
    shell: true
  });

  let output = '';
  let errorOutput = '';

  trainScript.stdout.on('data', (data) => {
    output += data.toString();
    console.log('训练输出:', data.toString());
  });

  trainScript.stderr.on('data', (data) => {
    errorOutput += data.toString();
    console.error('训练错误:', data.toString());
  });

  trainScript.on('close', (code) => {
    if (code === 0) {
      res.json({ success: true, message: '训练完成', output });
    } else {
      res.json({ success: false, message: '训练失败', error: errorOutput });
    }
  });
});

// 获取训练状态
app.get('/api/train/status', (req, res) => {
  const statusFile = path.join(__dirname, 'train_status.json');
  if (fs.existsSync(statusFile)) {
    const status = JSON.parse(fs.readFileSync(statusFile, 'utf8'));
    res.json(status);
  } else {
    res.json({ status: 'idle', progress: 0 });
  }
});

// 停止训练
app.post('/api/train/stop', (req, res) => {
  // 杀死训练进程
  spawn('pkill', ['-f', 'train.py']);
  res.json({ success: true, message: '训练已停止' });
});

// ============ 推理相关API ============

// 单张图片推理
app.post('/api/predict', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.json({ success: false, message: '请上传图片' });
  }

  const imagePath = req.file.path;
  console.log('推理图片:', imagePath);

  // 调用Python推理脚本
  const predictScript = spawn('python3', ['predict.py', imagePath], {
    cwd: __dirname,
    shell: true
  });

  let output = '';
  let errorOutput = '';

  predictScript.stdout.on('data', (data) => {
    output += data.toString();
  });

  predictScript.stderr.on('data', (data) => {
    errorOutput += data.toString();
    console.error('推理错误:', data.toString());
  });

  predictScript.on('close', (code) => {
    if (code === 0) {
      try {
        const result = JSON.parse(output);
        res.json({ 
          success: true, 
          result,
          resultImage: result.image ? `/results/${path.basename(result.image)}` : null
        });
      } catch (e) {
        res.json({ success: false, message: '解析结果失败', raw: output });
      }
    } else {
      res.json({ success: false, message: '推理失败', error: errorOutput });
    }
  });
});

// 批量推理
app.post('/api/predict/batch', upload.array('images', 10), (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.json({ success: false, message: '请上传图片' });
  }

  const imagePaths = req.files.map(f => f.path).join(',');
  console.log('批量推理图片:', req.files.length, '张');

  const predictScript = spawn('python3', ['predict.py', '--batch', imagePaths], {
    cwd: __dirname,
    shell: true
  });

  let output = '';
  let errorOutput = '';

  predictScript.stdout.on('data', (data) => {
    output += data.toString();
  });

  predictScript.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  predictScript.on('close', (code) => {
    if (code === 0) {
      try {
        const results = JSON.parse(output);
        res.json({ success: true, results });
      } catch (e) {
        res.json({ success: false, message: '解析结果失败' });
      }
    } else {
      res.json({ success: false, message: '推理失败' });
    }
  });
});

// 获取可用的训练好的模型
app.get('/api/models', (req, res) => {
  const modelsDir = path.join(__dirname, 'models');
  if (!fs.existsSync(modelsDir)) {
    return res.json({ models: [] });
  }
  
  const models = fs.readdirSync(modelsDir)
    .filter(f => f.endsWith('.pt') || f.endsWith('.pth'))
    .map(f => ({
      name: f,
      path: path.join(modelsDir, f),
      size: fs.statSync(path.join(modelsDir, f)).size
    }));
  
  res.json({ models });
});

// 上传预训练模型
app.post('/api/models', upload.single('model'), (req, res) => {
  if (!req.file) {
    return res.json({ success: false, message: '请上传模型文件' });
  }
  
  const modelPath = path.join(__dirname, 'models', req.file.filename);
  fs.renameSync(req.file.path, modelPath);
  
  res.json({ success: true, message: '模型上传成功', model: req.file.filename });
});

// ============ 文件下载 ============

// 下载推理结果
app.get('/api/download/:filename', (req, res) => {
  const filename = req.params.filename;
  const filePath = path.join(__dirname, 'public/results', filename);
  
  if (fs.existsSync(filePath)) {
    res.download(filePath);
  } else {
    res.status(404).json({ message: '文件不存在' });
  }
});

// 获取结果文件列表
app.get('/api/results', (req, res) => {
  const resultsDir = path.join(__dirname, 'public/results');
  if (!fs.existsSync(resultsDir)) {
    return res.json({ results: [] });
  }
  
  const files = fs.readdirSync(resultsDir)
    .filter(f => f.match(/\.(jpg|png|json)$/))
    .map(f => ({
      name: f,
      path: `/results/${f}`,
      created: fs.statSync(path.join(resultsDir, f)).mtime
    }));
  
  res.json({ results: files });
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`🚀 YOLO工业图像检测 running on http://localhost:${PORT}`);
});
