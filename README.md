# TASTE2-TW-QWEN2.5-Data

台灣繁體中文語音 SFT 資料生成 Pipeline。使用本地 Qwen2.5-32B-Instruct（vLLM）生成對話文字，再以 BreezyVoice（75%）/ IndexTTS-2（25%）合成音訊，最終輸出可直接餵入 TASTE2 SFT 訓練的 Parquet 格式。

---

## 資料類型

### Type 1 — User / Agent 對話（`user_agent`）

**情境**：使用者透過手機或電腦向 AI 助理提問，助理回答但不主動反問。

| 項目 | 規格 |
|------|------|
| 角色 | `User` / `Agent` |
| 每段對話輪數 | 8–14 輪（偶數） |
| 每輪最大字數 | 60 字 |
| 生成規模 | 18,755 段對話 |
| System Prompt | 一到兩句話，說明助理的領域專長與本次任務（每個情境獨立生成） |
| 主題 | 45 個主題（見下方主題列表） |
| SLURM Job | `slurm/tts_ua.job`, `slurm/tts_ua_remaining.job` |

**生成流程**：
1. 依主題生成情境描述（scenario）
2. 依情境生成 system prompt
3. 逐輪生成 User / Agent 發言

---

### Type 2 — Daily Conversation 日常閒聊（`daily_conv`）

**情境**：兩個普通人（朋友、同事、家人等）在日常場景中閒聊，非 user/agent 模式。

| 項目 | 規格 |
|------|------|
| 角色 | `甲` / `乙` |
| 每段對話輪數 | 8–14 輪（偶數） |
| 每輪最大字數 | 60 字 |
| 生成規模 | 27,285 段對話 |
| System Prompt | 描述兩人關係與當下場景（每個情境獨立生成） |
| 主題 | 45 個主題（見下方主題列表） |
| SLURM Job | `slurm/tts_dc.job`, `slurm/tts_dc_remaining.job` |

**生成流程**：
1. 依主題生成閒聊情境（兩人關係、地點、話題起因）
2. 逐輪生成 甲 / 乙 發言

---

### Type 3 — Instruction Following（`if_data`）

**情境**：甲連續發出 3–6 個語言指令，乙精準執行，涵蓋 14 種題型。

| 項目 | 規格 |
|------|------|
| 角色 | `User`（甲，發指令）/ `Agent`（乙，執行） |
| 每段對話輪數 | 任務數 × 2（每個任務一問一答） |
| 任務數/段 | 3–6 個（隨機） |
| 生成規模 | 13,076 段對話 |
| System Prompt | 固定：「兩人正在進行語言互動練習，一方發出指令，另一方精準執行各種任務。」 |
| 主題 | 無主題分類 |
| SLURM Job | `slurm/tts_if.job` |

#### 14 種題型與抽樣權重

| 題型 | 說明 | 抽樣權重 |
|------|------|----------|
| 朗讀 | 複誦指定句子 | 8 |
| 列舉 | 說出 N 種某類事物 | 4 |
| 數數 | 從 M 數到 N | 3 |
| 倒數 | 從 N 倒數到 M | 3 |
| 序列 | 說出固定序列（星期/月份/四季等） | 3 |
| 複述 | 將句子重複一遍（措辭可與朗讀不同） | 3 |
| 描述 | 場景或事物的短篇描述 | 3 |
| 問答 | 回答知識型問題 | 3 |
| 感受描述 | 描述某種感官或情緒感受 | 2 |
| 比較 | 比較兩樣事物哪個更… | 2 |
| 舉例 | 舉一個某類事物的例子 | 2 |
| 說出喜好 | 說出最喜歡／不喜歡的事物 | 2 |
| 反義詞 | 說出指定詞的反義詞 | 1 |
| 同義詞 | 說出指定詞的同義詞 | 1 |

題型由 LLM 依 few-shot 範例動態生成新題目，避免重複。有固定答案的題型（朗讀、複述、數數、倒數、序列、反義詞、同義詞、問答）乙必須答對。

---

## Parquet 欄位格式

每個 Parquet 檔案的 schema 如下：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `idx` | string | 唯一 ID，格式：`{type}_{topic}_{hash}` |
| `type` | string | `user_agent` / `daily_conv` / `if_data` |
| `topic` | string | 主題名稱（IF 為空） |
| `system_prompt` | string | 對話系統提示 |
| `meta` | string (JSON) | 含 `duration_sec`、`timestamp_range`、`speed`、`tts_backend` 等 |
| `message` | string (JSON) | 訊息列表（見下方） |

### `message` 欄位結構

```json
[
  {"role": "system",  "text": "<system_prompt>",   "audio": null},
  {"role": "User",    "text": "<turn_1_text>",      "audio": "<base64_wav>"},
  {"role": "Agent",   "text": "<turn_2_text>",      "audio": "<base64_wav>"},
  ...
]
```

- 第 0 筆為 system turn，`audio` 固定為 `null`
- 其餘每輪均有 `text`（繁體中文）與 `audio`（base64 編碼 WAV，16 kHz mono）
- DC 的角色名稱為 `甲` / `乙`

---

## 主題列表（45 個）

通用主題（41 個）：藝術、書籍、汽車、名人、程式設計、烹飪、教育、活動與展覽、時尚、健身、財經、美食、電玩遊戲、園藝、健康、歷史、嗜好、假期旅遊、居家生活、語言學習、彩妝、電影、音樂、大自然、新聞時事、寵物、哲學、攝影、播客節目、政治、人際關係、科學、購物、社群媒體、心靈成長、運動、科技、傳統文化、旅遊、天氣、工作職場

台灣在地主題（4 個）：台灣文化、台灣旅遊、台灣社會、台灣民俗信仰

---

## Pipeline 架構

```
conf/*.yaml          → 生成設定（主題、Prompt、輪數等）
generate/gen_*.py    → 文字生成（Qwen2.5-32B via vLLM）
tts/tts_runner.py    → TTS 合成（BreezyVoice 75% / IndexTTS-2 25%）
scripts/filter_english_dialogues.py  → 過濾含外文的對話
to_parquet/to_parquet_v5.py          → 轉換為 Parquet
```

## 快速開始

```bash
# 1. 啟動 vLLM server
bash vllm_server/launch_vllm.sh

# 2. 生成文字（以 user_agent 為例）
python generate/gen_user_agent.py --config conf/user_agent.yaml

# 3. TTS 合成
python tts/tts_runner.py --input output/user_agent/ --output tts_output/user_agent/

# 4. 轉 Parquet
python to_parquet/to_parquet_v5.py --input tts_output/user_agent/ --output parquet/user_agent/
```

大規模生成請使用 `slurm/` 下的 SLURM job scripts（64-GPU, 4 nodes）。

## 詳細規劃

見 [PLANNING.md](PLANNING.md)。
