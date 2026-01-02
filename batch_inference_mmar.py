
"""Flamingo CoT MMAR 批量推理脚本"""

import json
import os
import sys
import warnings
from datetime import datetime

# 禁用警告
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置工作目录
INFERENCE_DIR = "/mnt/afs/haizhouli-folder/interspeech/models/flamingo_cot_bp/af2/inference_HF_pretrained"
os.chdir(INFERENCE_DIR)
sys.path.insert(0, INFERENCE_DIR)

# 配置路径
QUESTIONS_FILE = "/mnt/afs/haizhouli-folder/data/MMAR_questions.json"
AUDIO_DIR = "/mnt/afs/haizhouli-folder/interspeech/MMAR-main/MMAR-main/audio"
OUTPUT_DIR = "/mnt/afs/haizhouli-folder/interspeech/Results"

# 导入必要模块
import torch
import yaml
import numpy as np
from src.factory import create_model_and_transforms
from utils import Dict2Class, get_autocast, get_cast_dtype
import librosa

# 加载配置
config = yaml.load(open("configs/inference.yaml"), Loader=yaml.FullLoader)
data_config = config['data_config']
clap_config = config['clap_config']  # 注意这里是 clap_config 不是 clap
inference_kwargs = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1
}

# 从 inference.py 复制的辅助函数
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def get_num_windows(T, sr, clap_config):
    window_length = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    
    num_windows = int((T - window_length) / (window_length - window_overlap)) + 1
    num_windows = min(num_windows, max_num_window)
    
    full_length = (num_windows - 1) * (window_length - window_overlap) + window_length
    return num_windows, full_length

def read_audio(audio_path, sr, duration, audio_start, clap_config):
    data_truncating = clap_config. get("data_truncating", "fusion")
    data_filling = clap_config.get("data_filling", "repeatpad")
    
    audio_time_series, _ = librosa.load(audio_path, sr=sr, offset=audio_start, duration=duration)
    
    if data_filling == "repeatpad":
        expected_length = int(duration * sr)
        if len(audio_time_series) < expected_length:
            n_repeat = int(expected_length / len(audio_time_series)) + 1
            audio_time_series = np.tile(audio_time_series, n_repeat)[:expected_length]
    
    data = audio_time_series
    if len(data) > 2:
        data = data / max(abs(data. max()), abs(data.min()))
    
    return data

def load_audio(audio_path, clap_config):
    sr = 16000
    window_length = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

    audio_data = read_audio(audio_path, sr, duration, 0.0, clap_config)
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data. reshape(1, -1)
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[: , start:start+window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) > max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)
    return audio_clips, audio_embed_mask

def load_model():
    """加载模型"""
    print("正在加载模型...")
    
    model, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="meta-llama/Llama-3.2-3B-Instruct",
        tokenizer_path="meta-llama/Llama-3.2-3B-Instruct",
        clap_encoder_config=clap_config,
        cross_attn_every_n_layers=1,
        use_local_files=True,
        decoder_layers_attr_name="model. layers",
        freeze_lm_embeddings=False,
        cache_dir="./safe_ckpt",
    )
    
    model.eval()
    device_id = torch.device("cuda" if torch.cuda. is_available() else "cpu")
    model = model.to(device_id)
    
    cast_dtype = get_cast_dtype(data_config['precision'])
    
    print("模型加载完成!\n")
    return model, tokenizer, device_id, cast_dtype

def predict(model, tokenizer, device_id, cast_dtype, filepath, question):
    """单个样本推理"""
    audio_clips, audio_embed_mask = load_audio(filepath, clap_config)
    audio_clips = audio_clips.to(device_id, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=cast_dtype, non_blocking=True)

    text_prompt = str(question).lower()
    sample = f"<audio>{text_prompt. strip()}{tokenizer.sep_token}"

    text = tokenizer(
        sample,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt"
    )

    input_ids = text["input_ids"].to(device_id, non_blocking=True)

    with torch.no_grad():
        output = model. generate(
            audio_x=audio_clips. unsqueeze(0),
            audio_x_mask=audio_embed_mask.unsqueeze(0),
            lang_x=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            **inference_kwargs,
        )[0]
    
    output_decoded = tokenizer.decode(output).split(tokenizer.sep_token)[-1]. replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')
    
    return output_decoded

def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"读取问题文件: {QUESTIONS_FILE}")
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    print(f"总共 {len(questions_data)} 个样本\n")
    
    model, tokenizer, device_id, cast_dtype = load_model()
    
    results = []
    
    for idx, item in enumerate(questions_data, 1):
        audio_id = item['id']
        question = item['question']
        audio_file = os.path.join(AUDIO_DIR, f"{audio_id}. wav")
        
        if not os.path.exists(audio_file):
            print(f"[{idx}/{len(questions_data)}] ⚠ 跳过:  {audio_id}")
            continue
        
        print(f"[{idx}/{len(questions_data)}] {audio_id}", end=" ...  ")
        
        try: 
            answer = predict(model, tokenizer, device_id, cast_dtype, audio_file, question)
            
            thinking_prediction = answer
            
            if '**' in answer:
                parts = answer.split('**')
                answer_prediction = parts[-1]. strip() if len(parts) > 1 else answer
            else:
                sentences = answer.split('.')
                answer_prediction = sentences[-1]. strip() if sentences else answer
            
            results.append({
                "id": audio_id,
                "thinking_prediction": thinking_prediction,
                "answer_prediction": answer_prediction
            })
            
            print("✓")
            
        except Exception as e:
            print(f"✗ {str(e)[:30]}")
            continue
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"flamingo_cot_results_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"推理完成！成功:  {len(results)}/{len(questions_data)}")
    print(f"结果保存至: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

