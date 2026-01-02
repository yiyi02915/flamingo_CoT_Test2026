#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MMAR CoT 推理脚本
输出格式: {id, thinking_prediction, answer_prediction}
"""

import os
import sys
import json
import yaml
import argparse
import re
from datetime import datetime
from pathlib import Path

# 添加代码路径
CODE_DIR = "/mnt/afs/haizhouli-folder/interspeech/models/flamingo_cot_bp/af2/inference_HF_pretrained"
sys.path.insert(0, CODE_DIR)

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.yaml")

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def format_question_with_choices(question, choices):
    """格式化问题，包含选项"""
    formatted = f"{question}\n\nChoices:\n"
    for i, choice in enumerate(choices, 1):
        formatted += f"{chr(64+i)}. {choice}\n"  # A, B, C, D
    formatted += "\nPlease think step by step and provide your final answer."
    return formatted

def extract_thinking_and_answer(output_text):
    """
    从模型输出中提取思考过程和最终答案
    """
    # 清理输出文本
    output_text = output_text.strip()
    
    # 尝试多种方式提取答案
    # 方式1: 查找 "Answer:" 或 "Final answer:" 等关键词
    answer_patterns = [
        r"(? : Final [Aa]nswer|Answer):\s*([A-D]\. ?\s*[^\n]+)",
        r"(?:The answer is|I choose|My answer is)\s*([A-D]\.?\s*[^\n]+)",
        r"\b([A-D])\.\s*([^\n]+)$",  # 最后一行的选项
    ]
    
    final_answer = ""
    for pattern in answer_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE | re.MULTILINE)
        if match:
            final_answer = match.group(1).strip()
            break
    
    # 如果没找到明确答案，尝试提取最后一句
    if not final_answer: 
        lines = output_text.strip().split('\n')
        if lines:
            final_answer = lines[-1].strip()
    
    # 思考过程就是完整输出
    thinking = output_text
    
    return thinking, final_answer

def run_inference_on_item(audio_path, question, config):
    """
    对单个音频运行推理并返回输出
    """
    # 创建临时 JSONL
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tf:
        temp_jsonl = tf.name
        data = {"audio_path": audio_path, "question": question}
        tf.write(json.dumps(data) + '\n')
    
    # 创建临时输出文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='. txt', delete=False) as of:
        temp_output = of.name
    
    try:
        # 切换到代码目录
        original_dir = os.getcwd()
        os.chdir(config['paths']['code_dir'])
        
        # 运行推理
        cmd = f"python inference.py --input {temp_jsonl} > {temp_output} 2>&1"
        ret_code = os.system(cmd)
        
        # 切换回原目录
        os.chdir(original_dir)
        
        # 读取输出
        with open(temp_output, 'r', encoding='utf-8', errors='replace') as f:
            output = f.read()
        
        return ret_code, output
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_jsonl):
            os.remove(temp_jsonl)
        if os.path.exists(temp_output):
            os.remove(temp_output)

def process_mmar_json(json_path, audio_base_dir, output_file, config, start_idx=0, end_idx=None, format_choices=True):
    """
    处理 MMAR JSON 文件并输出标准格式
    
    输出格式: 
    {
      "id": "<sample_id>",
      "thinking_prediction": "<model_generated_CoT>",
      "answer_prediction": "<final_prediction>"
    }
    """
    
    print("=" * 60)
    print("MMAR CoT 推理")
    print("=" * 60)
    
    # 读取 JSON
    print(f"\n加载 {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    print(f"✓ 总样本数: {total}")
    
    # 确定处理范围
    if end_idx is None:
        end_idx = total
    else:
        end_idx = min(end_idx, total)
    
    print(f"✓ 处理范围:  [{start_idx}, {end_idx})")
    print(f"✓ 将处理:  {end_idx - start_idx} 个样本")
    print(f"✓ 输出文件: {output_file}")
    print("=" * 60)
    
    # 统计
    success_count = 0
    error_count = 0
    
    # 结果列表
    results = []
    
    # 逐条处理
    for idx in range(start_idx, end_idx):
        item = data[idx]
        
        # 提取信息
        item_id = item['id']
        audio_rel_path = item['audio_path']
        question = item['question']
        choices = item.get('choices', [])
        
        # 构建完整音频路径
        audio_filename = os.path.basename(audio_rel_path)
        audio_full_path = os.path.join(audio_base_dir, audio_filename)
        
        # 检查音频文件
        if not os.path.exists(audio_full_path):
            error_msg = f"音频文件不存在: {audio_full_path}"
            print(f"[{idx+1}/{end_idx}] ❌ {item_id}: {error_msg}")
            
            # 记录错误结果
            results.append({
                "id": item_id,
                "thinking_prediction": f"ERROR: {error_msg}",
                "answer_prediction": "N/A"
            })
            error_count += 1
            continue
        
        # 格式化问题
        if format_choices and choices:
            full_question = format_question_with_choices(question, choices)
        else:
            full_question = question
        
        # 打印进度
        print(f"\n[{idx+1}/{end_idx}] 处理:  {item_id}")
        print(f"  音频: {audio_filename}")
        print(f"  问题: {question}")
        if choices:
            print(f"  选项:  {', '.join(choices)}")
        
        # 运行推理
        try:
            ret_code, output = run_inference_on_item(audio_full_path, full_question, config)
            
            if ret_code == 0:
                # 提取思考过程和答案
                thinking, answer = extract_thinking_and_answer(output)
                
                results.append({
                    "id": item_id,
                    "thinking_prediction": thinking,
                    "answer_prediction": answer
                })
                
                success_count += 1
                print(f"  ✓ 成功")
                print(f"  答案: {answer[: 100]}...")
                
            else:
                # 推理失败但有输出
                results.append({
                    "id": item_id,
                    "thinking_prediction": f"ERROR (code {ret_code}): {output}",
                    "answer_prediction": "N/A"
                })
                error_count += 1
                print(f"  ✗ 失败 (返回码: {ret_code})")
                print(f"  错误: {output[: 200]}...")
                
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            print(f"  ✗ 异常:  {error_msg}")
            
            results.append({
                "id": item_id,
                "thinking_prediction": f"EXCEPTION: {error_msg}",
                "answer_prediction": "N/A"
            })
        
        # 实时保存（每处理一条就保存一次，防止中断丢失）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 总结
    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    print(f"总计: {end_idx - start_idx}")
    print(f"成功:  {success_count}")
    print(f"失败: {error_count}")
    print(f"成功率: {success_count / (end_idx - start_idx) * 100:.1f}%")
    print(f"\n结果文件: {output_file}")
    print("=" * 60)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="MMAR CoT 推理（标准输出格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例: 
  # 测试前 5 个样本
  python run_mmar_cot.py --end 5 --output test_results.json
  
  # 处理全部样本
  python run_mmar_cot.py --output all_results.json
  
  # 处理指定范围
  python run_mmar_cot. py --start 100 --end 200 --output batch_results.json
        """
    )
    
    parser.add_argument('--config', type=str,
                       default=DEFAULT_CONFIG,
                       help='配置文件路径')
    parser.add_argument('--input', type=str,
                       default='/mnt/afs/haizhouli-folder/data/MMAR_questions. json',
                       help='MMAR JSON 文件路径')
    parser.add_argument('--audio-dir', type=str,
                       default='/mnt/afs/haizhouli-folder/interspeech/MMAR-main/MMAR-main/audio',
                       help='音频文件目录')
    parser.add_argument('--output', type=str,
                       required=True,
                       help='输出 JSON 文件路径')
    parser.add_argument('--start', type=int, default=0,
                       help='起始索引')
    parser.add_argument('--end', type=int, default=None,
                       help='结束索引（None=全部）')
    parser.add_argument('--no-format-choices', action='store_true',
                       help='不在问题中包含选项')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # 环境检查
    print("环境检查...")
    import torch
    print(f"✓ PyTorch:  {torch.__version__}")
    print(f"✓ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda. get_device_name(0)}")
    print()
    
    # 运行推理
    process_mmar_json(
        json_path=args.input,
        audio_base_dir=args.audio_dir,
        output_file=args.output,
        config=config,
        start_idx=args.start,
        end_idx=args.end,
        format_choices=not args.no_format_choices
    )

if __name__ == "__main__":
    main()

