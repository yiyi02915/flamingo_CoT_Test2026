#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试单个音频推理，查看详细错误
"""

import os
import sys
import json

# 添加代码路径
CODE_DIR = "/mnt/afs/haizhouli-folder/interspeech/models/flamingo_cot_bp/af2/inference_HF_pretrained"
sys.path.insert(0, CODE_DIR)

# 测试数据
audio_path = "/mnt/afs/haizhouli-folder/interspeech/MMAR-main/MMAR-main/audio/f0VchKwpMAk_00-11-10_00-11-30.wav"
question = """Determine what is producing the sound in the audio

Choices:
A.  Owl
B. Robot
C. Rooster
D.  Parrot

Please think step by step and provide your final answer."""

# 创建临时 JSONL
temp_jsonl = "/tmp/test_single.jsonl"
with open(temp_jsonl, 'w') as f:
    data = {"audio_path": audio_path, "question": question}
    f.write(json.dumps(data) + '\n')

print("=" * 60)
print("测试单个音频推理")
print("=" * 60)
print(f"音频: {audio_path}")
print(f"问题: {question[: 100]}...")
print(f"临时文件: {temp_jsonl}")
print("=" * 60)
print()

# 切换到代码目录并运行
os.chdir(CODE_DIR)
cmd = f"python inference.py --input {temp_jsonl}"
print(f"执行命令: {cmd}")
print("=" * 60)
print()

# 直接运行，不重定向输出，这样可以看到详细错误
os.system(cmd)

