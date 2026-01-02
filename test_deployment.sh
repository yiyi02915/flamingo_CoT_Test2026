#!/bin/bash

# Audio Flamingo 2 CoT 部署测试脚本

set -e

echo "=========================================="
echo "Audio Flamingo 2 CoT 部署测试"
echo "=========================================="

# 配置路径
DEPLOYMENT_DIR="/mnt/afs/haizhouli-folder/interspeech/deployment"
ENV_DIR="/mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot"

# 激活环境
echo "激活虚拟环境..."
source ${ENV_DIR}/bin/activate

# 测试 1: 环境检查
echo ""
echo "测试 1: 环境检查"
echo "------------------------------------------"
python ${DEPLOYMENT_DIR}/inference_server.py --mode check

# 测试 2: 创建示例文件
echo ""
echo "测试 2: 创建示例 JSONL 文件"
echo "------------------------------------------"
python ${DEPLOYMENT_DIR}/inference_server. py --mode sample

echo ""
echo "=========================================="
echo "✅ 基础测试完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 编辑 config.yaml，添加您的 HuggingFace Token"
echo "2. 准备音频文件和问题"
echo "3. 运行推理测试"
echo ""

