# DeepJiandu Glyph Detector

DeepJiandu 简牍文字框检测模型，用于在简牍图像中检测单字文字区域（单类：`text`）。

## 内容

- `models/deepjiandu-full-v1-best.pt`：全量版 v1 权重（约 5.91 MB）
- `scripts/yolo_glyph_predict.py`：单图推理脚本，输出 JSON 检测框
- `docs/MODEL_CARD.md`：模型卡、训练数据统计与评估指标
- `docs/validation-full-v1.md`：全量版 v1 训练与验证记录
- `requirements.txt`：推理依赖

## 快速使用

```powershell
python -m pip install -r requirements.txt
python scripts/yolo_glyph_predict.py path/to/slip.png --model models/deepjiandu-full-v1-best.pt
```

也可以通过 `GLYPH_YOLO_MODEL` 指定权重路径。脚本会自动选择 CUDA 或 CPU，并将检测框按从上到下、从左到右排序。

## 结果说明

在 DeepJiandu 自身验证集上，模型卡记录的指标为：mAP50 0.919、mAP50-95 0.534、精确率 0.912、召回率 0.861。模型只负责文字区域检测，不负责释文、字形识别或古文字释读。

## 数据与许可

本仓库不包含原始训练图片、数据集压缩包、PDF、数据库或其他未确认授权的研究资料。模型权重基于 DeepJiandu 公开数据集训练；使用者应分别遵守数据集、Ultralytics/YOLO 和输入图像的适用许可。当前模型仅作为研究用途发布，商业使用或再分发前请完成许可确认。

## 致谢

感谢 DeepJiandu 数据集及 Ultralytics YOLO 开源项目。
