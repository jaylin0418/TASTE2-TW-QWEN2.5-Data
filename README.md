# TASTE2-TW-QWEN2.5-Data

台灣繁體中文語音 SFT 資料生成 Pipeline。使用本地 Qwen2.5-32B-Instruct（vLLM）生成對話文字，再以 IndexTTS-2（50%）/ BreezyVoice（50%）合成音訊，最終輸出可直接餵入 TASTE2 SFT 訓練的 Parquet 格式。

---

## 資料類型

### Type 1 — User / Agent 對話（`user_agent`）

**情境**：使用者透過手機或電腦向 AI 助理提問，助理回答。

| 項目 | 規格 |
|------|------|
| 角色 | `User` / `Agent` |
| 每段對話輪數 | 8–14 輪（偶數） |
| 每輪最大字數 | 60 字 |
| 生成規模 | ~18,000 段對話 |
| 主題 | 45 個（見下方主題列表） |
| SLURM Job | `slurm/gen_text_array.job` → `slurm/tts_ua.job` → `slurm/to_parquet_ua.job` |

**範例**：
```
User：最近常頭痛，有什麼改善方法嗎？
Agent：可以試試多喝水、調整睡眠時間，以及減少看螢幕的時間。
User：睡眠時間要幾小時比較好？
Agent：成人一般建議七到八小時，規律的作息對緩解頭痛也有幫助。
```

---

### Type 2 — Daily Conversation 日常閒聊（`daily_conv`）

**情境**：兩個普通人（朋友、同事、家人等）在日常場景中閒聊。

| 項目 | 規格 |
|------|------|
| 角色 | `甲` / `乙` |
| 每段對話輪數 | 8–14 輪（偶數） |
| 每輪最大字數 | 60 字 |
| 生成規模 | ~27,000 段對話 |
| 主題 | 45 個（見下方主題列表） |
| SLURM Job | `slurm/gen_text_dc.job` → `slurm/tts_dc.job` → `slurm/to_parquet_dc.job` |

**範例**：
```
甲：欸你最近有在追什麼劇嗎？
乙：有啊，最近在看一個台灣偶像劇，劇情還不錯。
甲：什麼劇啊，說來聽聽？
乙：就是那個在台北拍的愛情劇，場景都很熟悉。
```

---

### Type 3 — Instruction Following（`if_data`）

**情境**：甲連續發出 3–6 個語言指令，乙精準執行，涵蓋 14 種題型。無速度控制。

| 項目 | 規格 |
|------|------|
| 角色 | `User`（甲，發指令）/ `Agent`（乙，執行） |
| 每段對話輪數 | 任務數 × 2 |
| 任務數/段 | 3–6 個（隨機） |
| 生成規模 | ~13,000 段對話 |
| 主題 | 無主題分類 |
| SLURM Job | `slurm/gen_if_array.job` → `slurm/tts_if_array.job` → `slurm/to_parquet_if.job` |

**範例**：
```
User：請用快的速度描述一個下雨天的校園。
Agent：校園裡雨水綿綿，樹葉閃亮，學生們撐著傘匆匆往教室走。
User：現在從十倒數到三。
Agent：十、九、八、七、六、五、四、三。
User：說出台灣的四季。
Agent：春天、夏天、秋天、冬天。
```

---

### Type 4 — IF with Speed Control（`if_control`）

**情境**：甲在每個語言任務裡明確指定速度（快速／慢速／正常），乙以對應速度執行。速度逐任務獨立指定，不 sticky。

| 項目 | 規格 |
|------|------|
| 角色 | `User`（甲，發指令）/ `Agent`（乙，執行） |
| 每段對話輪數 | 任務數 × 2 |
| 任務數/段 | 3–6 個（隨機） |
| 速度分布 | fast / slow / normal 各 1/3；normal 中一半會明說「正常速度」，一半不提 |
| 生成規模 | ~16,000 段對話（目標） |
| 主題 | 無主題分類 |
| SLURM Job | `slurm/gen_if_control.job` → `slurm/tts_if_control_array.job` → `slurm/to_parquet_if_control.job` |
| TTS | Agent turns 有 `speed` 欄位，fast/slow turns 音訊透過 `stretch_wav` 拉伸 |

**範例**：
```
User：用快的速度描述一個下雨天的校園。
Agent：[fast] 校園裡雨水綿綿，樹葉閃亮，學生們撐著傘匆匆往教室走。
User：放慢腳步說，把四季的順序唸出來。
Agent：[slow] 春天，夏天，秋天，冬天。
User：照正常速度，讀這句話，台北是一座繁忙的城市。
Agent：[normal] 台北是一座繁忙的城市。
User：加快速度，列出五種動物。
Agent：[fast] 老虎，獅子，大象，猴子，斑馬。
```

---

### Type 5 — Speed UA（`speed_ua`）

**情境**：User 與 Agent 多輪對話，User 透過語速請求控制 Agent 的說話速度。速度有**黏性（sticky）**，一旦切換後持續到下一次明確指令。

| 項目 | 規格 |
|------|------|
| 角色 | `User` / `Agent` |
| 每段對話輪數 | 8–14 輪（偶數） |
| 每輪最大字數 | 60 字 |
| 生成規模 | 45 主題 × 200 段 = ~9,000 段對話（目標） |
| 主題 | 45 個（見下方主題列表） |
| SLURM Job | `slurm/gen_speed_ua.job` → `slurm/tts_speed_ua_array.job` → `slurm/to_parquet_speed_ua.job` |
| TTS | User 永遠正常速度；Agent turns 有 `speed` 欄位，fast/slow 透過 `stretch_wav` 拉伸 |

**FSM 轉換概率**：
- `normal` → stay 60% / fast 20% / slow 20%
- `fast` → stay 62% / normal 33% / slow 5%
- `slow` → stay 62% / normal 33% / fast 5%

**速度注入規則**：
- 對話第一輪若為 normal：不提速度；若為 fast/slow：必須明說
- 轉換到任何狀態：User 必須在話裡自然帶入速度請求
- Sticky（持續同狀態）：User 完全不提速度

**範例**：
```
User：最近眼睛老是累得不得了，你有啥眼保健操可以教我嗎，說慢一點啦。
Agent：[slow] 可以做眼球轉動，上下左右各看三下，再閉目養神半分鐘。
User：這方法聽起來還真不錯，那要做幾遍會比較有效果呢？
Agent：[slow] 一天做三到四次，每次一套動作，應該會有幫助。
User：這樣啊，那如果時間真的很緊張，最少要做幾次才有用？
Agent：[slow] 最少每天也要做一兩次，還是要做完整套動作。
User：明白了，好你用正常速度說就好，那怎麼記得要做呢？
Agent：[normal] 可以用手機設定提醒，每隔一小時提醒休息一下眼睛。
```

---

## Parquet 欄位格式

每個 Parquet 檔案的 schema 如下：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `idx` | string | 唯一 dialogue ID |
| `type` | string | `type1_user_agent` / `type2_daily_conv` / `type3_if_data` / `type4_if_control` / `type5_speed_ua` |
| `topic` | string | 主題名稱（IF types 為空） |
| `system_prompt` | string | 對話系統提示 |
| `meta` | struct | `{data_type, tts_backend, if_tasks, speed_sequence}` |
| `message` | list[struct] | 訊息列表（見下方） |

### `message` 欄位結構

```json
[
  {"role": "system", "text": "<system_prompt>", "audio": null,        "timestamp_range": [], "speed": "normal"},
  {"role": "User",   "text": "<turn_text>",     "audio": "<wav_bytes>", "timestamp_range": [0, 3200], "speed": "normal"},
  {"role": "Agent",  "text": "<turn_text>",     "audio": "<wav_bytes>", "timestamp_range": [3450, 7100], "speed": "slow"}
]
```

- `audio`：WAV bytes（24 kHz mono）
- `timestamp_range`：`[start_ms, end_ms]`，相對於 full.wav
- `speed`：`"normal"` / `"fast"` / `"slow"`；User turns 永遠為 `"normal"`

---

## 主題列表（45 個）

通用（41）：藝術、書籍、汽車、名人、程式設計、烹飪、教育、活動與展覽、時尚、健身、財經、美食、電玩遊戲、園藝、健康、歷史、嗜好、假期旅遊、居家生活、語言學習、彩妝、電影、音樂、大自然、新聞時事、寵物、哲學、攝影、播客節目、政治、人際關係、科學、購物、社群媒體、心靈成長、運動、科技、傳統文化、旅遊、天氣、工作職場

台灣在地（4）：台灣文化、台灣旅遊、台灣社會、台灣民俗信仰

---

## Pipeline 架構

```
conf/*.yaml                              生成設定（主題、Prompt、輪數、速度等）
generate/gen_user_agent.py               Type 1 文字生成
generate/gen_daily_conv.py               Type 2 文字生成
generate/gen_if_data.py                  Type 3 文字生成
generate/gen_if_control.py              Type 4 文字生成
generate/gen_speed_ua.py                Type 5 文字生成
tts/tts_runner.py                        TTS 合成 + speed stretch
scripts/filter_english_dialogues.py     過濾含外文對話
to_parquet/to_parquet_v5.py             轉換為 Parquet
vllm_server/launch_vllm_local.sh        登入節點 vLLM 啟動（H100）
vllm_server/launch_vllm.job             SLURM vLLM job
```

## 快速開始

```bash
# 1. 啟動 vLLM server（登入節點 H100）
bash vllm_server/launch_vllm_local.sh

# 2. 品質預覽（確認生成品質後再大量生產）
conda run -n vllm_py312 python3 test_if_control.py   # Type 4
conda run -n vllm_py312 python3 test_speed_ua.py     # Type 5

# 3. 大規模文字生成（SLURM）
sbatch slurm/gen_if_control.job    # Type 4，array 0-7
sbatch slurm/gen_speed_ua.job      # Type 5，array 0-44 topics

# 4. TTS 合成（SLURM，需 GPU）
sbatch slurm/tts_if_control_array.job
sbatch slurm/tts_speed_ua_array.job

# 5. 轉 Parquet
sbatch slurm/to_parquet_if_control.job
sbatch slurm/to_parquet_speed_ua.job
```

## 詳細規劃

見 [PLANNING.md](PLANNING.md)。
