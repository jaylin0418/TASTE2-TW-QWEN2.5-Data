# TASTE2-TW-QWEN2.5-Data

台灣繁體中文語音 SFT 資料生成 Pipeline，基於本地 Qwen2.5-7B-Instruct（vLLM）生成文字，搭配 IndexTTS-2 / BreezyVoice 合成音訊。

## 目標

生成 1300 小時台灣在地化繁體中文語音資料，供 TASTE2 SFT 訓練使用。

## 資料分佈

| 類型 | 時數 | 說明 |
|------|------|------|
| User / Agent 對話 | 500h | 自然問答對話，助理不一直反問 |
| Daily Conversation | 500h | 兩人閒聊，非 user/agent 風格 |
| Instruction Following | 100h | 多個 IF 問題串成一段對話 |
| Speed Control | 100h | FSM 控制語速（fast / normal / slow） |
| IF Control | 100h | 使用者下語速指令，句型多樣化 |

## 詳細規劃

見 [PLANNING.md](PLANNING.md)。

## 快速開始

```bash
# 1. 啟動 vLLM server
bash vllm_server/launch_vllm.sh

# 2. 生成文字資料
python generate/gen_user_agent.py --config conf/user_agent.yaml

# 3. TTS 合成
python tts/tts_runner.py --input output/user_agent/ --output tts_output/user_agent/

# 4. 轉 Parquet
python to_parquet/to_parquet_v5.py --input tts_output/user_agent/ --output parquet/user_agent/
```
