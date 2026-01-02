#!/bin/bash

# Audio Flamingo 2 CoT 环境安装脚本
# 作者:  Copilot
# 日期: 2026-01-01

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Audio Flamingo 2 CoT 环境安装开始"
echo "=========================================="

# 配置路径
BASE_DIR="/mnt/afs/haizhouli-folder/interspeech"
ENV_DIR="${BASE_DIR}/model_env/flamingo_cot"
CODE_DIR="${BASE_DIR}/models/flamingo_cot_bp/af2/inference_HF_pretrained"
MODEL_DIR="${BASE_DIR}/models/flamingo_cot"

echo "环境目录: ${ENV_DIR}"
echo "代码目录: ${CODE_DIR}"
echo "模型目录: ${MODEL_DIR}"

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "当前 Python 版本: ${PYTHON_VERSION}"

# 创建虚拟环境（如果不存在）
if [ ! -d "${ENV_DIR}" ]; then
    echo "创建 Python 虚拟环境..."
    python3 -m venv ${ENV_DIR}
else
    echo "虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source ${ENV_DIR}/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 12.4 兼容版本)
echo "安装 PyTorch for CUDA 12.4..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
echo "安装 Audio Flamingo 2 依赖..."
cd ${CODE_DIR}
pip install -r requirements.txt

# 验证安装
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA 版本: {torch.version.cuda}')"
python -c "import torch; print(f'GPU 数量: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers 版本: {transformers.__version__}')"

echo ""
echo "=========================================="
echo "✅ 环境安装完成!"
echo "=========================================="
echo "激活环境命令:  source ${ENV_DIR}/bin/activate"
echo ""

