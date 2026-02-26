# 预训练权重

- `best_model_unsupervised.pt` / `unsupervised/`: 无监督模型
- `best_model_supervised.pt` / `supervised/`: 全监督模型

使用示例:
```bash
python live_predict_webcam.py --train-exp-dir pretrained/unsupervised --source 0 --duration 60
python live_predict_webcam.py --train-exp-dir pretrained/supervised --source video.mp4 --face
```
