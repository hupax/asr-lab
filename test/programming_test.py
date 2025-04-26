#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 测试FunASR模型对编程语音的识别能力

from funasr import AutoModel
import os
import sys

def test_programming_audio(audio_path, model_type="paraformer"):
    """
    测试不同模型对编程相关语音的识别能力
    """
    print(f"正在加载{model_type}模型...")

    if model_type == "paraformer":
        # Paraformer模型 - 适合中文识别
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
        )
    elif model_type == "whisper":
        # Whisper模型 - 适合多语言和专业术语识别
        model = AutoModel(
            model="iic/Whisper-large-v3-turbo",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
        )
    else:
        print(f"不支持的模型类型: {model_type}")
        return

    print(f"模型加载完成，开始识别音频: {audio_path}")

    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件 {audio_path} 不存在!")
        return

    # 根据不同模型类型设置不同参数
    if model_type == "paraformer":
        res = model.generate(
            input=audio_path,
            batch_size_s=300,
        )
    else:  # whisper
        DecodingOptions = {
            "task": "transcribe",
            "language": None,  # 可以设置为特定语言如"zh"或"en"
            "beam_size": None,
            "fp16": True,
            "without_timestamps": False,
            "prompt": None,
        }
        res = model.generate(
            DecodingOptions=DecodingOptions,
            batch_size_s=0,
            input=audio_path,
        )

    print("\n===== 识别结果 =====")
    print(res)

    # 保存结果到文件
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 提取文件名作为输出文件名的一部分
    base_name = os.path.basename(audio_path)
    file_name = os.path.splitext(base_name)[0]

    output_file = os.path.join(output_dir, f"{file_name}_{model_type}_result.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"音频文件: {audio_path}\n")
        f.write(f"模型类型: {model_type}\n\n")
        f.write("识别结果:\n")
        f.write(str(res))

    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python programming_test.py <音频文件路径> [模型类型]")
        print("模型类型可选值: paraformer(默认), whisper")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "paraformer"

    test_programming_audio(audio_path, model_type)
