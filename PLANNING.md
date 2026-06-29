# Pipeline V5 — 規劃文件

**目標：** 生成 1300 小時繁體中文（台灣在地化）語音資料，供 TASTE2 SFT 訓練使用。

---

## 一、總體架構

### LLM 後端：Qwen2.5-7B-Instruct（本地 vLLM）

- **模型路徑：** `/work/jaylin0418/home_models/Qwen2.5-7B-Instruct/`
- **推論方式：** vLLM，OpenAI-compatible HTTP API（`http://localhost:8000/v1`）
- **GPU：** H100 NVL（95GB），`tensor_parallel_size` 視可用卡數而定
- **原則：** 完全不使用 OpenRouter，全部走本地 vLLM

### TTS 後端

| 資料類型 | TTS 後端 | Speed Control |
|----------|----------|---------------|
| User/Agent（500h） | IndexTTS-2 × BreezyVoice 各半 | 無 |
| Daily Conversation（500h） | IndexTTS-2 × BreezyVoice 各半 | 無 |
| Instruction Following（100h） | IndexTTS-2 × BreezyVoice 各半 | 無 |
| Speed Control（100h） | BreezyVoice + time-stretching | ✅ fast / normal / slow |
| IF Control（100h） | BreezyVoice + time-stretching | ✅ fast / normal / slow |

Speed control 只用 `audiostretchy` 做 time-stretching，**不加任何 emotion tag**。
唯一的 TTS-level tag 為 `speed: fast` 或 `speed: slow`（normal 不加 tag）。

### 語言規範（全部資料）

- **完全不出現英文**（包括 system prompt、scenario、dialogue 所有部分）
- 所有外來語用繁體中文表達（例：人工智慧、網際網路、咖啡廳）
- 所有數字若涉及朗讀，用國字（例：一、二、三）
- **強調台灣在地化資訊**，包括但不限於：
  - 地標：台北一〇一、九份、阿里山、日月潭、墾丁、花蓮太魯閣
  - 交通：捷運、台鐵、高鐵、公車
  - 飲食：珍珠奶茶、鹹酥雞、蚵仔煎、夜市小吃
  - 習俗：媽祖遶境、元宵燈會、中秋烤肉
  - 媒體/品牌：誠品、全聯、7-11

---

## 二、資料分佈與時數估算

### 時數換算假設

| 參數 | 數值 |
|------|------|
| 每回合最大字元數 | 60 字（→ 約 20 秒語音） |
| 每段對話回合數 | 12–16 輪 |
| 每段對話音訊時長 | 約 4–5 分鐘 |
| 平均使用 4.5 分鐘 |

| 類型 | 目標時數 | 需生對話數 |
|------|----------|------------|
| User / Agent | 500 h | ~6,700 段 |
| Daily Conversation | 500 h | ~6,700 段 |
| Instruction Following | 100 h | ~1,340 段 |
| Speed Control | 100 h | ~1,340 段 |
| IF Control | 100 h | ~1,340 段 |
| **合計** | **1300 h** | **~17,400 段** |

---

## 三、各類型資料規格

### 3.1 User / Agent 對話（500h）

**特色：**
- 兩人對話，有明確的「使用者」與「助理」角色
- 助理自然回應，**不一直反問**，有時主動分享資訊、說故事、給建議
- 話題涵蓋 41 個主題（沿用現有主題清單，全翻成繁中）
- 無任何聲學控制標記

**對話風格指引（寫進 system prompt）：**
- 助理有自己的想法與觀點，不一昧順著使用者
- 助理偶爾分享個人經驗（虛構），讓對話更自然
- 避免「您有什麼問題嗎？」「請問還有其他需要嗎？」等客服口吻
- 適當使用台灣口語：「對啊」「真的假的」「超～的」「然後啊」

**LLM Prompt 結構：**
```
1. Scenario 生成（單一 LLM call）→ 台灣在地化場景描述
2. System Prompt 生成 → 針對此場景的助理角色設定
3. Dialogue 生成 → 雙角色交替，12–16 輪，每輪 ≤ 60 字
```

---

### 3.2 Daily Conversation（500h）

**特色：**
- 兩個普通人（非 user/agent）的日常閒聊
- 沒有「幫助解決問題」的目的性，就是純聊天
- 角色為朋友、同事、鄰居、家人等
- 語氣更隨意，有時候說到一半換話題

**對話風格指引：**
- 可以用台灣日常用語、口頭禪
- 話題可以自然跳躍（聊食物聊到旅遊再聊到家人）
- 允許相對短的回應（「嗯嗯」「是哦」「哈哈真的假的」）
- 偶爾插入提到台灣地點/食物/節日

**LLM Prompt 結構：**
```
1. 場景生成 → 兩人關係、當下情境（例：同事午休、朋友逛夜市）
2. Dialogue 生成 → 無角色限制，自然閒聊，12–16 輪，每輪 ≤ 60 字
```

---

### 3.3 Instruction Following（100h）

**特色：**
- 一個人問多個問題串成一段對話（非單輪 QA）
- 每段對話包含 3–6 個 IF 類型問題，自然串接
- 問題類別參考 `pipeline_v4_IF/generate_if_data_zh.py` 的 25 個類別
- **不加任何 speed / emotion 控制**

**25 個 IF 類別（全繁中）：**
```
朗讀、數數、序列、倒序、列舉、重複、拼音/注音、數字朗讀、
時間日期朗讀、格式限制、否定限制、必要詞彙、詞彙抽取、
替換、過濾、選擇、排序、比較、完成句子、轉換、
短篇描述、短篇生成、簡單計算、條件推理、多步驟推理
```

**對話結構：**
```
使用者：[IF 問題 1]
助理：[回答 1]
使用者：[IF 問題 2（自然銜接）]
助理：[回答 2]
... 重複 3–6 輪
```

---

### 3.4 Speed Control（100h）

**特色：**
- 基本對話結構（User/Agent 或 Daily 皆可）
- 對話中的語速由 FSM 控制，每輪可能切換
- **只控制語速**，無 emotion control

**FSM 規格（Finite State Machine）：**

```
狀態：Normal / Fast / Slow

轉移規則：
  Normal → Fast：機率 P_start（第一次轉移機率高）
  Normal → Slow：機率 P_start（第一次轉移機率高）
  Fast   → Normal：機率 P_return
  Fast   → Fast（維持）：1 - P_return
  Slow   → Normal：機率 P_return
  Slow   → Slow（維持）：1 - P_return

建議預設值（待確認）：
  P_start_first = 0.5   # 第一次離開 Normal 的機率
  P_start_later = 0.15  # 之後再次切換的機率（回 Normal 後）
  P_return = 0.4        # 從 Fast/Slow 回 Normal 的機率
```

**TTS 處理：**
- `speed: fast` → time-stretch factor `0.77`（加速約 30%）
- `speed: slow` → time-stretch factor `1.33`（減速約 33%）
- Normal → 不做 time-stretch

**對話中的 Speed Tag 標記方式（純文字層面，不出現在語音）：**
每輪對話的 metadata 記錄當前 speed 狀態，TTS 時讀取並處理。

---

### 3.5 IF Control（100h）

**特色：**
- 使用者「下指令」要求用特定語速說話，然後助理照做
- 指令句型多樣化（LLM 自動生成，不要每次都一樣）
- **只控制語速（fast / slow）**，也有要求恢復正常的指令

**指令句型範例（LLM 要自動生成更多變化）：**
```
語速指令（快）：
  「你可以說快一點嗎？」
  「講話節奏能不能快一些？」
  「拜託說話不要這麼慢」
  「你說話速度可以加快嗎？」

語速指令（慢）：
  「你能說慢一點嗎？」
  「可以放慢你說話的速度嗎？」
  「你說太快了，能不能慢一些？」
  「請說慢一點，我沒跟上」

恢復正常：
  「好了，不用這麼慢了」
  「可以說正常速度了」
  「沒關係，正常說就好」
```

**對話結構：**
```
... 幾輪正常對話 ...
使用者：[speed 控制指令]
助理：[確認 + 用新速度回應]   ← time-stretch 開始
... 幾輪保持新速度 ...
使用者：[恢復正常 or 換速度]
助理：[確認 + 照做]
```

**FSM 參考 Speed Control（3.4），但觸發方式是使用者說出指令句**

---

## 四、程式架構設計

```
pipeline_v5/
├── README.md                    ← 此文件
├── conf/
│   ├── user_agent.yaml
│   ├── daily_conv.yaml
│   ├── if_data.yaml
│   ├── speed_control.yaml
│   └── if_control.yaml
├── vllm_server/
│   ├── launch_vllm.sh           ← 啟動 vLLM server
│   └── check_vllm.py            ← 健康檢查
├── generate/
│   ├── base_generator.py        ← 共用 LLM client（對接 vLLM）
│   ├── gen_user_agent.py        ← Type 1
│   ├── gen_daily_conv.py        ← Type 2
│   ├── gen_if_data.py           ← Type 3
│   ├── gen_speed_control.py     ← Type 4（含 FSM）
│   └── gen_if_control.py        ← Type 5
├── tts/
│   ├── tts_runner.py            ← 統一 TTS 介面（IndexTTS-2 / BreezyVoice）
│   └── speed_stretch.py         ← audiostretchy time-stretching
├── to_parquet/
│   └── to_parquet_v5.py         ← 統一轉 Parquet
├── slurm/
│   ├── launch_vllm.job          ← 申請 GPU 跑 vLLM
│   ├── gen_text_array.job       ← 文字生成 array job
│   └── tts_array.job            ← TTS 合成 array job
└── run_pipeline.sh              ← 一鍵執行腳本（Dev 用）
```

---

## 五、vLLM 設定

```bash
# 模型下載位置
MODEL_PATH=/work/jaylin0418/home_models/Qwen2.5-7B-Instruct

# 啟動指令（以 2 張 H100 為例）
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --port 8000

# Python 呼叫（OpenAI-compatible）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")
```

---

## 六、待確認事項（Open Questions）

以下問題需要確認後才能開始實作：

### Q1：對話長度
目前規劃每輪 ≤ 60 字、12–16 輪、約 4.5 分鐘/段。
**是否調整？** 例如 Daily Conversation 是否允許更短的輪次（更碎片化）？

### Q2：主題設定
- User/Agent 和 Daily Conversation 是否沿用現有 41 個主題（全換成繁中）？
- 還是要新增更多台灣在地主題（例：廟宇文化、台灣選舉、夜市攤販）？

### Q3：Speed FSM 機率參數
規劃的預設值：
- `P_start_first = 0.5`（第一次從 Normal 跳走的機率）
- `P_start_later = 0.15`（之後每輪繼續切換的機率）
- `P_return = 0.4`（從 Fast/Slow 回 Normal 的機率）
**是否認可這些數值？**

### Q4：vLLM GPU 數量
目前機器只看到 1 張 H100 NVL。
- **是否透過 SLURM 申請多張？** 需要幾張做 tensor parallel？
- 或者 1 張跑 Qwen2.5-7B 就夠（7B 在 95GB 完全放得下）？

### Q5：IndexTTS-2 與 BreezyVoice 各半的分法
「各一半」是指：
- (A) 每個 sample 50% 機率選一個後端？
- (B) 總體資料集 50% 用 IndexTTS-2、50% 用 BreezyVoice（按 topic 或 batch 分）？

### Q6：IF Control 的 FSM 設計
IF Control（Type 5）的速度切換，是沿用 Type 4 的 FSM（由程式決定何時切），
還是改為「完全由 LLM 生成對話，LLM 自己決定何時下指令」？

### Q7：每個主題要生幾段對話？
以 User/Agent 為例，6700 段 ÷ 41 主題 ≈ 163 段/主題，是否合理？
或者某些主題要更多（例如 Food、Travel），某些要更少？

---

## 七、實作順序規劃

1. **[ ] 拉 Qwen2.5-7B-Instruct 模型**（`huggingface-cli download`）
2. **[ ] 寫 vLLM 啟動腳本 + SLURM job**
3. **[ ] 寫 `base_generator.py`**（vLLM OpenAI client 封裝）
4. **[ ] 實作 Type 1 User/Agent**（最基礎，確認 pipeline 可跑通）
5. **[ ] 實作 Type 2 Daily Conversation**（修改 prompt 即可）
6. **[ ] 實作 Type 3 IF Data**（多 IF 串接）
7. **[ ] 實作 Speed FSM + TTS time-stretch**（Type 4）
8. **[ ] 實作 Type 5 IF Control**（在 FSM 上加指令句生成）
9. **[ ] 統一 to_parquet 轉換**
10. **[ ] SLURM array job 設定（大規模生產）**
