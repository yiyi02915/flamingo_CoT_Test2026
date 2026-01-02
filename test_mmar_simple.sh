#!/bin/bash

# MMAR 简单测试脚本（直接读取 JSON）

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "MMAR 直接推理测试"
echo -e "==========================================${NC}"

# 配置
ENV_ROOT="/mnt/afs/haizhouli-folder/interspeech/model_env/flamingo_cot"
DEPLOY_DIR="${ENV_ROOT}/deploy"

# 激活环境
echo -e "${YELLOW}激活环境...${NC}"
source ${ENV_ROOT}/bin/activate
cd ${DEPLOY_DIR}

# 环境检查
echo -e "\n${BLUE}步骤 1: 环境检查${NC}"
python inference_server.py --mode check

# 测试推理（前 5 个样本）
echo -e "\n${BLUE}步骤 2: 测试推理（前 5 个样本）${NC}"
python run_mmar_direct.py --end 5

echo -e "\n${GREEN}=========================================="
echo "✅ 测试完成！"
echo -e "==========================================${NC}"

echo -e "\n${YELLOW}下一步: ${NC}"
echo "1. 检查测试结果:"
echo "   ls -lh /mnt/afs/haizhouli-folder/interspeech/Results/"
echo ""
echo "2. 查看结果内容:"
echo "   cat /mnt/afs/haizhouli-folder/interspeech/Results/mmar_results_*. jsonl"
echo ""
echo "3. 运行完整推理:"
echo "   python run_mmar_direct.py"
echo ""
echo "4. 或指定范围:"
echo "   python run_mmar_direct.py --start 0 --end 100"
echo ""

