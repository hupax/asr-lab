#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: pip3 install -U openai-whisper

from funasr import AutoModel
import re

model = AutoModel(
    model="iic/Whisper-large-v3-turbo",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
)

# 修复解码选项，移除不支持的参数
DecodingOptions = {
    "task": "transcribe",     # 使用转写任务
    "language": "zh",         # 明确指定中文，更好地处理标点
    "beam_size": 5,           # 增加beam search宽度，提高识别质量
    "fp16": True,
    "without_timestamps": False,
    "prompt": None,
    "suppress_blank": False,  # 不抑制空白，保留原始输出
    # 移除了不支持的 word_timestamps 和 suppress_tokens 参数
}

res = model.generate(
    DecodingOptions=DecodingOptions,
    batch_size_s=0,
    input="/Users/hupax/FunASR/test/techcpm.wav",
)

# 打印完整的结果以查看所有信息
print("完整结果:")
print(res)

# 只打印文本部分
if res and len(res) > 0 and "text" in res[0]:
    print("\n识别文本:")
    print(res[0]["text"])

# 使用标点恢复模型处理
if res and len(res) > 0 and "text" in res[0]:
    try:
        print("\n使用标点恢复模型处理...")
        punc_model = AutoModel(model="ct-punc")
        text = res[0]["text"]
        punc_res = punc_model.generate(input=text)

        if punc_res and len(punc_res) > 0 and "text" in punc_res[0]:
            punc_text = punc_res[0]["text"]

            # 后处理：修复技术术语被错误分割的问题
            print("\n修复技术术语...")

            # 常见技术术语列表（可根据需要扩展）
            tech_terms = [
                "Node.js", "Nodejs", "JavaScript", "TypeScript", "React", "Vue", "Angular",
                "Cloudflare", "Worker", "Warsail", "Netlify", "GitHub", "GitLab", "BitBucket",
                "VSCode", "Visual Studio Code", "OpenRouter", "API", "SEO", "next.js", "Next.js",
                "Reactor", "React Native", "Serverless", "Function", "AI", "Client"
            ]

            # 修复被空格和标点分割的技术术语
            fixed_text = punc_text
            for term in tech_terms:
                # 创建可能的分割模式（考虑字母间可能有空格、点、逗号等）
                pattern = r''.join([c + r'[\s\.\,、]*' for c in term[:-1]]) + term[-1]
                # 查找并替换
                matches = re.finditer(pattern, fixed_text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group(0)
                    if matched_text != term and (re.search(r'[\s\.\,、]', matched_text)):
                        fixed_text = fixed_text.replace(matched_text, term)

            # 处理特殊情况
            # 修复 "S erv erles sF unction" 类型的分割
            fixed_text = re.sub(r'S\s*erv\s*erles\s*s\s*F\s*unction', 'Serverless Function', fixed_text)
            # 修复 "next . js" 类型的分割
            fixed_text = re.sub(r'next\s*\.\s*js', 'next.js', fixed_text)
            # 修复 "N ode . js" 类型的分割
            fixed_text = re.sub(r'N\s*ode\s*\.\s*js', 'Node.js', fixed_text)
            # 修复 "R ea ctor N ative" 类型的分割
            fixed_text = re.sub(r'R\s*ea\s*ctor\s*N\s*ative', 'React Native', fixed_text)
            # 修复 "R ea ctor" 类型的分割
            fixed_text = re.sub(r'R\s*ea\s*ctor', 'React', fixed_text)
            # 修复 "C loud flare W ork er" 类型的分割
            fixed_text = re.sub(r'C\s*[\,\s]*loud\s*[\,\s]*flare', 'Cloudflare', fixed_text)
            fixed_text = re.sub(r'W\s*[\,\s]*ork\s*[\,\s]*er', 'Worker', fixed_text)
            fixed_text = re.sub(r'W\s*[\,\s]*arsa\s*[\,\s]*il', 'Warsail', fixed_text)
            fixed_text = re.sub(r'N\s*[\,\s]*etli\s*[\,\s]*fy', 'Netlify', fixed_text)
            # 修复 "VSC ode" 类型的分割
            fixed_text = re.sub(r'VSC\s*ode', 'VSCode', fixed_text)
            fixed_text = re.sub(r'VS\s*C\s*ode', 'VSCode', fixed_text)
            # 修复 "O pen R outer" 类型的分割
            fixed_text = re.sub(r'O\s*pen\s*R\s*outer', 'OpenRouter', fixed_text)
            # 修复 "C lien t" 类型的分割
            fixed_text = re.sub(r'C\s*lien\s*t', 'Client', fixed_text)
            # 修复地址
            fixed_text = re.sub(r'code\s*\.\s*vs\s*hows\s*studio\s*\.\s*com', 'code.vshowstudio.com', fixed_text)

            print("\n添加标点并修复技术术语后的文本:")
            print(fixed_text)
    except Exception as e:
        print(f"标点恢复处理出错: {e}")
