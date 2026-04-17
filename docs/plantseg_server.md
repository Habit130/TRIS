# PlantSeg Linux Server Runbook

## 目录约定

- 仓库目录：`TRIS/`
- 数据目录：`../plantseg/`
- 训练权重：`weights/plantseg/stage2/`
- 最优 checkpoint 固定别名：`weights/plantseg/stage2/best_model.pth`
- 官方 warm start：`weights/pretrained/stage2_refcocog_umd.pth`
- 测试 mask 输出：`output/plantseg/test_masks/ann/`
- 测试指标摘要：`output/plantseg/eval/test_metrics.json`

## 环境

- 使用 `environment.server.plantseg.yml`
- 目标环境固定为 Linux + 单卡 RTX 4090 + CUDA 11.8 + Python 3.10 + Miniconda

## 数据要求

- `../plantseg/main.json` 必须存在
- `main.json` 的 `image` 与 `mask` 字段必须继续指向 `../plantseg/` 下的相对路径
- 训练 / 验证 / 测试划分直接使用 `main.json` 中的 `split`
- 文本固定使用 `caption[3]`

## 训练

- 入口脚本：`scripts/train_plantseg.sh`
- 训练主线：`train_stage2.py --dataset plantseg`
- 模型会自动下载：
  - OpenAI 官方 CLIP RN50 权重
  - 官方 `stage2_refcocog_umd.pth` warm start

## 评测

- 入口脚本：`scripts/validate_plantseg.sh`
- 默认读取：`weights/plantseg/stage2/best_model.pth`
- 评测脚本：`validate_plantseg.py`
- 输出指标：
  - `IoU`
  - `Dice`
  - `Recall`
  - `mIoU`
  - `mACC`
- `test` split 的预测 mask 会保存成与真值同名、同尺寸、同单通道 PNG 二值格式
