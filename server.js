const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = 6002;

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use('/downloads', express.static(path.join(__dirname, 'public/results')));
app.use('/videos', express.static(path.join(__dirname, 'public/videos')));

// 确保目录存在
const dirs = ['public/uploads', 'public/results', 'public/videos', 'datasets/ok', 'datasets/ng', 'models', 'temp'];
dirs.forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Multer配置 - 上传图片
const imageStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'public/uploads');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});
const uploadImage = multer({ storage: imageStorage });

// Multer配置 - 上传压缩包
const zipStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'temp');
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  }
});
const uploadZip = multer({ storage: zipStorage });

// Multer配置 - 上传视频
const videoStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'public/videos');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});
const uploadVideo = multer({ storage: videoStorage });

// 任务状态存储
const jobs = {};

// ============ 训练相关API ============

// 上传OK图片压缩包
app.post('/api/train/dataset/ok', uploadZip.single('file'), async (req, res) => {
  if (!req.file) {
    return res.json({ success: false, message: '请上传压缩包' });
  }

  const zipPath = req.file.path;
  const extractDir = path.join(__dirname, 'datasets', 'ok');

  // 解压
  spawn('unzip', ['-o', zipPath, '-d', extractDir]).on('close', (code) => {
    fs.unlinkSync(zipPath); // 删除zip文件
    if (code === 0) {
      // 统计图片数量
      const files = fs.readdirSync(extractDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f));
      res.json({ success: true, message: `OK数据集上传成功，共 ${files.length} 张图片` });
    } else {
      res.json({ success: false, message: '解压失败' });
    }
  });
});

// 上传NG图片压缩包
app.post('/api/train/dataset/ng', uploadZip.single('file'), async (req, res) => {
  if (!req.file) {
    return res.json({ success: false, message: '请上传压缩包' });
  }

  const zipPath = req.file.path;
  const extractDir = path.join(__dirname, 'datasets', 'ng');

  // 解压
  spawn('unzip', ['-o', zipPath, '-d', extractDir]).on('close', (code) => {
    fs.unlinkSync(zipPath);
    if (code === 0) {
      const files = fs.readdirSync(extractDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f));
      res.json({ success: true, message: `NG数据集上传成功，共 ${files.length} 张图片` });
    } else {
      res.json({ success: false, message: '解压失败' });
    }
  });
});

// 获取数据集状态
app.get('/api/train/dataset', (req, res) => {
  const okDir = path.join(__dirname, 'datasets', 'ok');
  const ngDir = path.join(__dirname, 'datasets', 'ng');

  const okCount = fs.existsSync(okDir) ? fs.readdirSync(okDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f)).length : 0;
  const ngCount = fs.existsSync(ngDir) ? fs.readdirSync(ngDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f)).length : 0;

  res.json({ 
    ok: okCount, 
    ng: ngCount,
    ready: okCount > 0 && ngCount > 0
  });
});

// 开始训练
app.post('/api/train/start', async (req, res) => {
  const { modelType, epochs, batchSize, imageSize } = req.body;
  
  const okDir = path.join(__dirname, 'datasets', 'ok');
  const ngDir = path.join(__dirname, 'datasets', 'ng');
  
  if (!fs.existsSync(okDir) || !fs.existsSync(ngDir)) {
    return res.json({ success: false, message: '请先上传OK和NG数据集' });
  }

  const jobId = uuidv4();
  const config = {
    jobId,
    modelType: modelType || 'yolov8n',
    epochs: parseInt(epochs) || 50,
    batchSize: parseInt(batchSize) || 8,
    imageSize: parseInt(imageSize) || 640
  };

  jobs[jobId] = { status: 'training', progress: 0, message: '准备训练...' };
  console.log('开始训练，配置:', config);

  // 生成YOLO格式数据集
  const generateScript = spawn('python3', ['generate_dataset.py', JSON.stringify(config)], {
    cwd: __dirname,
    shell: true
  });

  generateScript.stdout.on('data', (data) => {
    console.log('数据集生成:', data.toString());
  });

  generateScript.stderr.on('data', (data) => {
    console.error('错误:', data.toString());
  });

  generateScript.on('close', async (code) => {
    if (code === 0) {
      // 开始训练
      jobs[jobId].message = '数据集生成完成，开始训练...';
      
      const trainScript = spawn('python3', ['train.py', JSON.stringify(config)], {
        cwd: __dirname,
        shell: true
      });

      trainScript.stdout.on('data', (data) => {
        const output = data.toString();
        console.log('训练输出:', output);
        
        // 解析进度
        const match = output.match(/Epoch\s+(\d+)\/(\d+)/);
        if (match) {
          const progress = Math.round((parseInt(match[1]) / parseInt(match[2])) * 100);
          jobs[jobId].progress = progress;
          jobs[jobId].message = `训练中: ${match[1]}/${match[2]}`;
        }
      });

      trainScript.stderr.on('data', (data) => {
        console.error('训练错误:', data.toString());
      });

      trainScript.on('close', (trainCode) => {
        if (trainCode === 0) {
          jobs[jobId].status = 'completed';
          jobs[jobId].progress = 100;
          jobs[jobId].message = '训练完成!';
          res.json({ success: true, jobId, message: '训练完成' });
        } else {
          jobs[jobId].status = 'error';
          jobs[jobId].message = '训练失败';
          res.json({ success: false, message: '训练失败' });
        }
      });
    } else {
      jobs[jobId].status = 'error';
      jobs[jobId].message = '数据集生成失败';
      res.json({ success: false, message: '数据集生成失败' });
    }
  });
});

// 获取训练状态
app.get('/api/train/status/:jobId', (req, res) => {
  const { jobId } = req.params;
  const job = jobs[jobId];
  if (job) {
    res.json(job);
  } else {
    res.json({ status: 'not_found' });
  }
});

// ============ 推理相关API ============

// 单张图片推理
app.post('/api/predict', uploadImage.single('image'), (req, res) => {
  if (!req.file) {
    return res.json({ success: false, message: '请上传图片' });
  }

  const imagePath = req.file.path;
  console.log('推理图片:', imagePath);

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
        res.json({ success: false, message: '解析结果失败' });
      }
    } else {
      res.json({ success: false, message: '推理失败' });
    }
  });
});

// ============ 视频Demo相关API ============

// 生成视频Demo
app.post('/api/demo/video', uploadVideo.single('video'), async (req, res) => {
  if (!req.file) {
    return res.json({ success: false, message: '请上传视频' });
  }

  const videoPath = req.file.path;
  const jobId = uuidv4();
  
  // 查找最新的训练模型
  const modelsDir = path.join(__dirname, 'models');
  let modelPath = '';
  
  if (fs.existsSync(modelsDir)) {
    const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.pt'));
    if (files.length > 0) {
      // 按修改时间排序
      files.sort((a, b) => {
        return fs.statSync(path.join(modelsDir, b)).mtime - fs.statSync(path.join(modelsDir, a)).mtime;
      });
      modelPath = path.join(modelsDir, files[0]);
    }
  }

  if (!modelPath) {
    return res.json({ success: false, message: '没有训练好的模型，请先训练模型' });
  }

  jobs[jobId] = { status: 'processing', progress: 0, message: '开始处理视频...' };
  
  console.log('生成视频Demo:', videoPath, '模型:', modelPath);

  const outputName = `demo_${Date.now()}.mp4`;
  const outputPath = path.join(__dirname, 'public/videos', outputName);

  const demoScript = spawn('python3', ['demo.py', videoPath, modelPath, outputPath], {
    cwd: __dirname,
    shell: true
  });

  demoScript.stdout.on('data', (data) => {
    const output = data.toString();
    console.log('Demo输出:', output);
    
    const match = output.match(/(\d+)%/);
    if (match) {
      jobs[jobId].progress = parseInt(match[1]);
      jobs[jobId].message = `处理中: ${match[1]}`;
    }
  });

  demoScript.stderr.on('data', (data) => {
    console.error('Demo错误:', data.toString());
  });

  demoScript.on('close', (code) => {
    if (code === 0) {
      jobs[jobId].status = 'completed';
      jobs[jobId].progress = 100;
      jobs[jobId].message = '处理完成';
      jobs[jobId].output = `/videos/${outputName}`;
      res.json({ success: true, jobId, output: `/videos/${outputName}` });
    } else {
      jobs[jobId].status = 'error';
      jobs[jobId].message = '处理失败';
      res.json({ success: false, message: '视频处理失败' });
    }
  });
});

// 获取Demo状态
app.get('/api/demo/status/:jobId', (req, res) => {
  const { jobId } = req.params;
  const job = jobs[jobId];
  if (job) {
    res.json(job);
  } else {
    res.json({ status: 'not_found' });
  }
});

// ============ 模型管理 ============

// 获取可用的训练好的模型
app.get('/api/models', (req, res) => {
  const modelsDir = path.join(__dirname, 'models');
  if (!fs.existsSync(modelsDir)) {
    return res.json({ models: [] });
  }
  
  const models = fs.readdirSync(modelsDir)
    .filter(f => f.endsWith('.pt'))
    .map(f => ({
      name: f,
      path: path.join(modelsDir, f),
      size: fs.statSync(path.join(modelsDir, f)).size,
      mtime: fs.statSync(path.join(modelsDir, f)).mtime
    }))
    .sort((a, b) => b.mtime - a.mtime);
  
  res.json({ models });
});

// ============ 文件下载 ============

// 下载视频
app.get('/api/video/download/:filename', (req, res) => {
  const filename = req.params.filename;
  const filePath = path.join(__dirname, 'public/videos', filename);
  
  if (fs.existsSync(filePath)) {
    res.download(filePath);
  } else {
    res.status(404).json({ message: '文件不存在' });
  }
});

// 获取视频列表
app.get('/api/videos', (req, res) => {
  const videosDir = path.join(__dirname, 'public/videos');
  if (!fs.existsSync(videosDir)) {
    return res.json({ videos: [] });
  }
  
  const files = fs.readdirSync(videosDir)
    .filter(f => f.match(/\.(mp4|avi|mov)$/))
    .map(f => ({
      name: f,
      path: `/videos/${f}`,
      size: fs.statSync(path.join(videosDir, f)).size,
      created: fs.statSync(path.join(videosDir, f)).mtime
    }))
    .sort((a, b) => b.created - a.created);
  
  res.json({ videos: files });
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`🚀 YOLO工业图像检测 running on http://localhost:${PORT}`);
});
