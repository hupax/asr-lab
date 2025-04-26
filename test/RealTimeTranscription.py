from funasr import AutoModel
import soundfile
import numpy as np
import pyaudio
import time
import os

# 配置参数
chunk_size = [0, 10, 5] # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 # number of encoder chunks to lookback for decoder cross-attention
sample_rate = 16000 # 采样率
channels = 1 # 单声道
format = pyaudio.paInt16 # 采样位数
chunk_samples = chunk_size[1] * 960 # 每次处理的音频采样点数 (600ms)

# 初始化语音识别模型和标点恢复模型
print("正在加载语音识别和标点模型...")
# 语音识别模型
asr_model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
# 标点恢复模型
punc_model = AutoModel(model="ct-punc", model_revision="v2.0.4")
print("模型加载完成，开始录音...")

# 初始化PyAudio
p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(
    format=format,
    channels=channels,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_samples
)

# 用于保存所有识别结果
all_recognition_results = []
all_recognition_results_with_punc = []
full_text = ""
full_text_with_punc = ""
last_text = ""
punc_cache = {}

# 实时识别处理
try:
    cache = {}
    print("开始实时语音识别，按Ctrl+C停止...")

    while True:
        # 读取音频数据
        audio_data = stream.read(chunk_samples)

        # 转换为numpy数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 使用ASR模型进行识别
        res = asr_model.generate(
            input=audio_array,
            cache=cache,
            is_final=False,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back
        )

        # 如果有识别结果则打印
        if res and len(res) > 0 and 'text' in res[0]:
            current_text = res[0]['text'].strip()
            if current_text and current_text != last_text:
                # 使用标点恢复模型处理
                punc_res = punc_model.generate(input=current_text, cache=punc_cache)
                current_text_with_punc = punc_res[0]['text'] if punc_res and len(punc_res) > 0 and 'text' in punc_res[0] else current_text

                print(f"实时识别: {current_text}")
                print(f"添加标点: {current_text_with_punc}")

                last_text = current_text

                # 将新结果添加到列表和完整文本中
                all_recognition_results.append(current_text)
                all_recognition_results_with_punc.append(current_text_with_punc)

                if full_text:
                    full_text += " " + current_text
                else:
                    full_text = current_text

                if full_text_with_punc:
                    full_text_with_punc += " " + current_text_with_punc
                else:
                    full_text_with_punc = current_text_with_punc

        # 简单的控制识别速率
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n停止录音和识别")
finally:
    # 关闭并终止
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 清理最后一次识别
    try:
        res = asr_model.generate(
            input=np.zeros(1, dtype=np.float32),
            cache=cache,
            is_final=True,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back
        )
        if res and len(res) > 0 and 'text' in res[0]:
            final_text = res[0]['text'].strip()
            if final_text and final_text != last_text:
                # 使用标点恢复模型处理
                punc_res = punc_model.generate(input=final_text)
                final_text_with_punc = punc_res[0]['text'] if punc_res and len(punc_res) > 0 and 'text' in punc_res[0] else final_text

                print(f"最终识别: {final_text}")
                print(f"添加标点: {final_text_with_punc}")

                all_recognition_results.append(final_text)
                all_recognition_results_with_punc.append(final_text_with_punc)

                full_text += " " + final_text
                full_text_with_punc += " " + final_text_with_punc
    except:
        pass

    # 对整个文本进行一次标点恢复，以获得更连贯的效果
    try:
        final_punc_res = punc_model.generate(input=full_text)
        if final_punc_res and len(final_punc_res) > 0 and 'text' in final_punc_res[0]:
            final_full_text_with_punc = final_punc_res[0]['text']
        else:
            final_full_text_with_punc = full_text_with_punc
    except:
        final_full_text_with_punc = full_text_with_punc

    # 输出所有识别结果
    print("\n===== 完整识别记录 =====")
    for i, (text, text_with_punc) in enumerate(zip(all_recognition_results, all_recognition_results_with_punc), 1):
        print(f"{i}. 原始: {text}")
        print(f"   标点: {text_with_punc}")

    print("\n===== 完整识别文本（无标点）=====")
    print(full_text)

    print("\n===== 完整识别文本（带标点）=====")
    print(final_full_text_with_punc)

    print("\n程序已退出")
