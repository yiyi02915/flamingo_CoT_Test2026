#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Flamingo 2 CoT 推理服务脚本
支持 MMAR 数据集推理
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加代码路径
CODE_DIR = "/mnt/afs/haizhouli-folder/interspeech/models/flamingo_cot_bp/af2/inference_HF_pretrained"
sys.path.insert(0, CODE_DIR)

# 获取当前脚本所在目录
SCRIPT_DIR = os.path. dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.yaml")

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def check_environment(config):
    """检查环境配置"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    import torch
    print(f"✓ PyTorch 版本: {torch.__version__}")
    print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA 版本: {torch.version.cuda}")
        print(f"✓ GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\n路径检查:")
    print(f"✓ 工作目录: {config['paths']['env_root']}")
    print(f"✓ 模型目录: {config['paths']['model_dir']}")
    print(f"✓ 代码目录: {config['paths']['code_dir']}")
    
    # 检查 MMAR 数据路径
    input_json = config['paths']['input_json']
    audio_dir = config['paths']['audio_dir']
    output_dir = config['paths']['output_dir']
    
    print(f"\nMMAR 数据路径:")
    if os.path.exists(input_json):
        print(f"✓ 输入 JSON:  {input_json}")
    else:
        print(f"✗ 输入 JSON 不存在:  {input_json}")
    
    if os.path.exists(audio_dir):
        num_audio = len([f for f in os.listdir(audio_dir) if os.path.isfile(os. path.join(audio_dir, f))])
        print(f"✓ 音频目录: {audio_dir} ({num_audio} 个文件)")
    else:
        print(f"✗ 音频目录不存在:  {audio_dir}")
    
    if os.path.exists(output_dir):
        print(f"✓ 输出目录: {output_dir}")
    else:
        print(f"!  输出目录不存在，将自动创建: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)

def run_inference_batch(input_jsonl, config, save_results=True):
    """批量推理"""
    print(f"\n{'='*60}")
    print("批量推理")
    print(f"{'='*60}")
    print(f"输入文件: {input_jsonl}")
    
    # 检查输入文件
    if not os.path.exists(input_jsonl):
        print(f"❌ 错误:  输入文件不存在:  {input_jsonl}")
        return
    
    # 统计任务数量
    with open(input_jsonl, 'r') as f:
        num_tasks = sum(1 for _ in f)
    print(f"✓ 任务数量: {num_tasks}")
    
    # 准备输出
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(config['paths']['output_dir'], f"results_{timestamp}.jsonl")
        print(f"✓ 结果将保存到: {output_file}")
    
    # 调用原始推理脚本
    os.chdir(config['paths']['code_dir'])
    cmd = f"python inference. py --input {input_jsonl}"
    
    if save_results:
        cmd += f" > {output_file}. log 2>&1"
    
    print(f"\n执行命令:  {cmd}\n")
    print(f"{'='*60}\n")
    
    # 运行推理
    ret = os.system(cmd)
    
    print(f"\n{'='*60}")
    if ret == 0:
        print("✅ 批量推理完成")
        if save_results:
            print(f"✓ 日志文件: {output_file}.log")
            print(f"✓ 结果目录: {config['paths']['output_dir']}")
    else:
        print(f"❌ 推理出错，返回码: {ret}")
    print(f"{'='*60}")

def run_inference_single(audio_path, question, config):
    """单个音频推理"""
    print(f"\n{'='*60}")
    print("单音频推理")
    print(f"{'='*60}")
    print(f"音频文件:  {audio_path}")
    print(f"问题: {question}")
    
    # 检查音频文件
    if not os.path.exists(audio_path):
        print(f"❌ 错误:  音频文件不存在: {audio_path}")
        return
    
    # 创建临时 JSONL 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_jsonl = f"/tmp/inference_{timestamp}.jsonl"
    
    with open(temp_jsonl, 'w') as f:
        data = {"audio_path": audio_path, "question": question}
        f.write(json.dumps(data) + '\n')
    
    print(f"✓ 临时文件: {temp_jsonl}")
    
    # 调用原始推理脚本
    os.chdir(config['paths']['code_dir'])
    cmd = f"python inference. py --input {temp_jsonl}"
    print(f"\n执行命令: {cmd}\n")
    print(f"{'='*60}\n")
    
    os.system(cmd)
    
    # 清理临时文件
    if os.path.exists(temp_jsonl):
        os.remove(temp_jsonl)
    
    print(f"\n{'='*60}")
    print("✅ 推理完成")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Audio Flamingo 2 CoT 推理服务 - MMAR 版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例: 
  # 环境检查
  python inference_server.py --mode check
  
  # 单个音频推理
  python inference_server. py --mode single \\
    --audio /mnt/afs/haizhouli-folder/interspeech/MMAR-main/MMAR-main/audio/example.wav \\
    --question "描述这段音频"
  
  # 批量推理（使用准备好的 JSONL）
  python inference_server.py --mode batch \\
    --input ../workspace/examples/mmar_tasks. jsonl
        """
    )
    
    parser.add_argument('--config', type=str,
                       default=DEFAULT_CONFIG,
                       help='配置文件路径')
    parser.add_argument('--mode', type=str,
                       choices=['single', 'batch', 'check'],
                       default='check',
                       help='运行模式')
    parser.add_argument('--audio', type=str, help='音频文件路径 (single模式)')
    parser.add_argument('--question', type=str,
                       default='Describe the audio in detail.',
                       help='问题 (single模式)')
    parser.add_argument('--input', type=str, help='输入 JSONL 文件 (batch模式)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果到文件')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    if args.mode == 'check':
        # 环境检查模式
        check_environment(config)
        
    elif args.mode == 'single':
        # 单个音频推理
        if not args.audio:
            print("❌ 错误: single 模式需要指定 --audio 参数")
            parser.print_help()
            return
        check_environment(config)
        run_inference_single(args.audio, args.question, config)
        
    elif args.mode == 'batch':
        # 批量推理
        if not args.input:
            print("❌ 错误: batch 模式需要指定 --input 参数")
            parser.print_help()
            return
        check_environment(config)
        run_inference_batch(args.input, config, save_results=not args.no_save)

if __name__ == "__main__":
    main()

