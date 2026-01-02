# Audio Flamingo 2 CoT 部署指南

本文档提供 Audio Flamingo 2 CoT 模型的完整部署和使用说明。

## 📋 目录结构

```
/mnt/afs/haizhouli-folder/interspeech/
├── models/
│   ├── flamingo_cot/              # 模型权重文件
│   └── flamingo_cot_bp/           # 源代码
│       └── af2/
│           └── inference_HF_pretrained/
├── model_env/
│   └── flamingo_cot/              # Python 虚拟环境
├── deployment/                     # 部署脚本 (本目录)
│   ├── setup_env.sh               # 环境安装脚本
│   ├── config.yaml                # 配置文件
│   ├── inference_server.py        # 推理服务脚本
│   ├── test_deployment.sh         # 测试脚本
│   └── README_DEPLOY.md           # 本文档
└── output/                         # 输出目录 (自动创建)
```

## 🚀 快速开始

### 步骤 1: 安装环境

```bash
cd /mnt/afs/haizhouli-folder/interspeech/deployment
bash setup_env.sh
```

安装时间约 10-15 分钟，根据网络速度而定。

### 步骤 2: 配置 HuggingFace Token

1. 获取 HuggingFace Token:  https://huggingface.co/settings/tokens
2. 编辑配置文件: 

```bash
vi config.yaml
```

3. 将 `YOUR_HUGGINGFACE_TOKEN_HERE` 替换为您的实际 Token

**或者直接修改推理脚本:**

```bash
vi /mnt/afs/haizhouli-folder/interspeech/models/flamingo_cot_bp/af2/inference_HF_pretrained/inference. py
```

在第 183 行替换您的 Token。

### 步骤 3: 测试部署

```bash
# 激活环境
source /mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot/bin/activate

# 运行测试
bash test_deployment.sh
```

## 📖 使用方法

### 方法 1: 环境检查

```bash
source /mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot/bin/activate
python inference_server.py --mode check
```

### 方法 2: 单个音频推理

```bash
source /mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot/bin/activate

python inference_server.py \
    --mode single \
    --audio /path/to/your/audio.wav \
    --question "Describe the audio in detail."
```

### 方法 3: 批量推理

1. 准备 JSONL 文件 (每行一个 JSON 对象):

```json
{"audio_path": "/path/to/audio1.wav", "question": "What sounds are in this audio?"}
{"audio_path": "/path/to/audio2.wav", "question": "Describe the music. "}
```

2. 运行批量推理:

```bash
source /mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot/bin/activate

python inference_server.py \
    --mode batch \
    --input your_questions.jsonl
```

### 方法 4: 创建示例文件

```bash
python inference_server.py --mode sample --output sample.jsonl
```

## 🔧 配置说明

### config.yaml 主要参数

```yaml
model:
  repo_id: "nvidia/audio-flamingo-2"  # 可选 3B/1. 5B/0.5B
  hf_token: "YOUR_TOKEN"

inference:
  temperature: 0.0      # 0.0=确定性, 0.7-1.0=创造性
  top_k:  50
  top_p:  0.95
  max_new_tokens: 512

hardware:
  device: "cuda"
  gpu_id: 0
  precision: "fp16"     # 或 "fp32"
```

### 模型版本选择

| 模型 | 参数量 | HuggingFace ID | 性能 |
|------|--------|----------------|------|
| 默认 | 3B | nvidia/audio-flamingo-2 | 最佳 |
| 中等 | 1.5B | nvidia/audio-flamingo-2-1.5B | 良好 |
| 小型 | 0.5B | nvidia/audio-flamingo-2-0.5B | 快速 |

## 📝 JSONL 文件格式

```json
{"audio_path": "/absolute/path/to/audio.wav", "question": "Your question here"}
```

**要求:**
- 使用绝对路径
- 支持格式:  WAV, MP3, FLAC
- 最大时长: 5 分钟
- 每行一个完整的 JSON 对象

## 🎯 示例问题

### 通用理解
- "Describe the audio in detail."
- "What sounds can you hear?"
- "Summarize this audio clip."

### 音乐分析
- "What is the genre of this music?"
- "Describe the instruments used."
- "What is the mood of this music?"

### 语音分析
- "What is the speaker's emotion?"
- "How many speakers are there?"
- "What is being discussed?"

### 环境音
- "Where was this audio recorded?"
- "What activities are happening?"
- "Describe the acoustic environment."

## ⚙️ 高级用法

### 直接使用原始推理脚本

```bash
cd /mnt/afs/haizhouli-folder/interspeech/models/flamingo_cot_bp/af2/inference_HF_pretrained

source /mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot/bin/activate

# 编辑 inference.jsonl
vi inference.jsonl

# 运行推理
python inference.py --input inference.jsonl
```

### 修改采样参数

编辑 `inference.py` 第 232 行: 

```python
# 确定性输出 (用于基准测试)
temperature=0.0, do_sample=False

# 或创造性输出 (用于对话)
temperature=0.8, top_k=50, top_p=0.95, do_sample=True
```

### 切换模型版本

编辑 `inference.py` 第 183 行和 `configs/inference.yaml` 第 81-82 行:

**0.5B 模型:**
```python
# inference.py L183
repo_id="nvidia/audio-flamingo-2-0.5B"

# configs/inference.yaml L81-82
lm_path:  Qwen/Qwen2.5-0.5B
lm_tokenizer_path: Qwen/Qwen2.5-0.5B
```

**1.5B 模型:**
```python
# inference.py L183
repo_id="nvidia/audio-flamingo-2-1.5B"

# configs/inference.yaml L81-82
lm_path: Qwen/Qwen2.5-1.5B
lm_tokenizer_path: Qwen/Qwen2.5-1.5B
```

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案:**
- 切换到较小的模型 (1.5B 或 0.5B)
- 减少音频时长
- 使用 `precision: "fp32"` 如果 fp16 有问题

### 2. HuggingFace Token 错误

**错误信息:** `401 Unauthorized`

**解决方案:**
- 确认 Token 有效:  https://huggingface.co/settings/tokens
- 确认有模型访问权限
- 修改 `inference.py` 第 183 行

### 3. 音频文件找不到

**解决方案:**
- 使用绝对路径
- 检查文件权限
- 确认音频格式 (WAV/MP3/FLAC)

### 4. 模块导入错误

**解决方案:**
```bash
# 重新激活环境
source /mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot/bin/activate

# 检查安装
pip list | grep torch
pip list | grep transformers
```

## 📊 性能优化

### H100 GPU 优化设置

您的 H100 80GB 非常强大，建议: 

```yaml
hardware:
  precision: "fp16"  # 使用混合精度
  
inference:
  # 可以处理更长的音频
  max_audio_duration: 300  # 5分钟
  max_new_tokens: 1024     # 更长的回答
```

### 批量处理优化

对于大量音频文件: 

```bash
# 并行处理多个 GPU (如果有多张卡)
CUDA_VISIBLE_DEVICES=0 python inference. py --input batch1.jsonl &
CUDA_VISIBLE_DEVICES=1 python inference.py --input batch2.jsonl &
```

## 📚 参考资料

- **论文:** https://arxiv.org/abs/2503.03983
- **项目主页:** https://github.com/NVIDIA/audio-flamingo
- **Demo:** https://research.nvidia.com/labs/adlr/AF2/
- **HuggingFace:** https://huggingface.co/nvidia/audio-flamingo-2

## 📄 许可证

- **代码:** MIT License
- **模型:** NVIDIA OneWay Noncommercial License (仅供非商业研究使用)
- **依赖:** Qwen Research License, OpenAI Terms of Use

## 💡 技术支持

如遇到问题: 

1. 检查环境:  `python inference_server.py --mode check`
2. 查看日志输出
3. 参考官方文档:  https://github.com/NVIDIA/audio-flamingo

---

**部署脚本作者:** GitHub Copilot  
**创建日期:** 2026-01-01  
**版本:** 1.0

