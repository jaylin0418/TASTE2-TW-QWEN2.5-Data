# Pipeline V5 — 規劃文件（TASTE2-TW-QWEN2.5-Data）

**目標：** 生成 1400 小時繁體中文（台灣在地化）語音資料，供 TASTE2 SFT 訓練使用。

---

## 一、總體架構

### LLM 後端：Qwen2.5-7B-Instruct（本地 vLLM）

- **模型路徑：** `/work/jaylin0418/home_models/Qwen2.5-7B-Instruct/`
- **推論方式：** vLLM，OpenAI-compatible HTTP API
- **GPU：** H100 NVL，透過 SLURM 申請 32 或 64 張，**完全平行化**（每個 worker 連同一個 vLLM endpoint）
- **原則：** 不使用 OpenRouter，全部走本地 vLLM

### TTS 後端

每個 topic 的前半 scenario 用 **IndexTTS-2**，後半用 **BreezyVoice**（由 scenario index 決定，不隨機）。

| 後端 | 聲音選取方式 |
|------|------------|
| IndexTTS-2 | User 和 Agent 各自固定一個說話者（每個 topic 的 speaker assignment 記錄在 JSON） |
| BreezyVoice | 隨機從 Mozilla Common Voice zh-TW 選聲音（與 v1_tc 做法相同） |

Speed control：在 TTS 後用 `audiostretchy` 做 time-stretching，**不加 emotion tag**。
每一輪都記錄 speed tag（寫入 metadata）：
- `speed: fast` → stretch factor `0.77`（加速 ～30%）
- `speed: slow` → stretch factor `1.33`（減速 ～33%）
- `speed: normal` → 不做 stretch，但仍記錄 tag

### 語言規範（全部資料）

- **完全不出現英文**（system prompt、scenario、dialogue 全部）
- 所有外來語一律用繁體中文（人工智慧、網際網路、咖啡廳、漢堡）
- **強調台灣在地化**，每段對話都要融入台灣脈絡：
  - 地標：台北一〇一、九份、阿里山、日月潭、墾丁、太魯閣、西門町
  - 交通：捷運、台鐵、高鐵、公車、YouBike
  - 飲食：珍珠奶茶、鹹酥雞、蚵仔煎、滷肉飯、夜市小吃、雞排
  - 習俗：媽祖遶境、元宵燈會、中秋烤肉、尾牙
  - 品牌/日常：誠品、全聯、統一超商、夜市、廟口

---

## 二、資料分佈與時數估算

### 對話長度規格

| 參數 | 數值 |
|------|------|
| 回合數 | 8–16 輪（每次隨機取 8～16 的偶數） |
| 每輪最大字元 | 60 字，**但每輪 TTS 後不超過 30 秒** |
| 口語化輪次 | 允許部分輪次非常短（如「對啊」「嗯嗯」「哈哈真的假的」） |
| System Prompt | 每段對話都有（參考 tc_emo 設計方式） |

### 時數估算

平均約 4 分鐘/段（含短輪次）：

| 類型 | 目標時數 | 生成段數目標 |
|------|----------|------------|
| User / Agent（無速度控制） | 500 h | ≥ 8,000 段 |
| Daily Conversation（無速度控制） | 500 h | ≥ 8,000 段 |
| Instruction Following | 100 h | ≥ 1,600 段 |
| User / Agent + Speed Control（FSM） | 100 h | ≥ 1,600 段 |
| Daily Conversation + Speed Control（FSM） | 100 h | ≥ 1,600 段 |
| IF Control（速度 + IF 指令串接） | 100 h | ≥ 1,600 段 |
| **合計** | **1400 h** | **≥ 22,400 段** |

寧願多生，不要少生。

---

## 三、主題清單

沿用 41 個原有主題（全改繁中台灣用語），另新增台灣在地主題。

### 原有主題（繁中化）

藝術、書籍、汽車、名人、程式設計、烹飪、教育、活動、時尚、健身、
財經、美食、電玩遊戲、園藝、健康、歷史、嗜好、假期旅遊、居家生活、
語言學習、彩妝、電影、音樂、大自然、新聞時事、寵物、哲學、攝影、
播客節目、政治、人際關係、科學、購物、社群媒體、心靈成長、運動、
科技、傳統文化、旅遊、天氣、工作職場

### 新增台灣在地主題

- 台灣夜市文化
- 台灣廟宇與民俗信仰
- 台灣選舉與公民社會
- 台灣景點與旅遊（花蓮、墾丁、九份、阿里山）
- 台灣美食（小吃、辦桌、珍奶文化）
- 台灣交通（捷運、高鐵日常）

---

## 四、各類型資料規格

### 4.1 User / Agent 對話（500h）

**特色：**
- 兩人對話（使用者 ↔ 助理），有各自固定角色
- 助理**不一直反問**，有時主動分享資訊、說故事、給建議、有自己的意見
- 話題涵蓋上方全部主題
- 每段有 system prompt（說明助理角色與對話情境）
- 無任何聲學控制

**助理風格要點（寫入 system prompt）：**
- 有自己的想法，不一昧順著使用者說
- 偶爾分享虛構個人經驗讓對話自然
- 避免客服口吻（「請問還有什麼需要協助？」）
- 適當使用台灣口語：「對啊」「真的假的」「超～的」「然後啊」

---

### 4.2 Daily Conversation（500h）

**特色：**
- 兩個普通人閒聊（朋友、同事、鄰居、家人），**非 user/agent 風格**
- 沒有「幫助解決問題」的目的性
- 語氣隨意，話題可以自然跳躍
- 允許非常短的回應（「嗯嗯」「是哦」「哈哈真的假的」）
- 每段有 system prompt（說明兩人關係與當下情境）

**場景範例：**
同事午休聊天、朋友逛夜市、家人吃飯、鄰居偶遇、LINE 語音通話

---

### 4.3 Instruction Following（100h）

**特色：**
- 一段對話內串接 3–6 個 IF 問題（自然銜接，不是單輪 QA）
- **不加任何速度或情緒控制**，所有輪次 `speed: normal`
- A 給指令、B 執行，類似單純的指令執行情境
- 問題類別（全繁中，25 種）：

```
朗讀、數數、序列、倒序、列舉、重複、注音拼讀、數字朗讀、
時間日期朗讀、格式限制、否定限制、必要詞彙、詞彙抽取、
替換、過濾、選擇、排序、比較、完成句子、轉換、
短篇描述、短篇生成、簡單計算、條件推理、多步驟推理
```

**對話結構：**
```
甲：請念以下這句話：台灣的珍珠奶茶聞名世界。
乙：台灣的珍珠奶茶聞名世界。
甲：接著，請從一數到五。
乙：一、二、三、四、五。
甲：好，現在列出三種台灣夜市小吃。
乙：鹹酥雞、蚵仔煎、滷肉飯。
```

---

### 4.4 User/Agent + Speed Control（100h）＆ 4.5 Daily Conv + Speed Control（100h）

**特色：**
- 對話結構分別同 Type 1（User/Agent）和 Type 2（Daily Conversation）
- 對話中語速由 **FSM 自動切換**（說話者不說出任何速度指令，速度只反映在 TTS 合成上）
- 每一輪記錄 speed tag
- 只控制語速，無 emotion

**FSM 規格：**

```
狀態：Normal（N）/ Fast（F）/ Slow（S）

初始狀態：Normal

轉移機率：
  ┌──────────────────────────────────────────────────────────┐
  │ 第一次離開 Normal（從未切換過）：P_leave_first = 0.75   │
  │   → Fast 或 Slow 各 50%                                 │
  ├──────────────────────────────────────────────────────────┤
  │ Fast / Slow → Normal：P_return = 0.40                   │
  │ Fast / Slow → 維持：1 - P_return = 0.60                 │
  ├──────────────────────────────────────────────────────────┤
  │ 回到 Normal 後再次切換：P_leave_again = 0.40            │
  │（比第一次低，但仍有相當機率繼續切換）                    │
  └──────────────────────────────────────────────────────────┘

每輪對話結束後執行一次 FSM 轉移。
速度標記寫入 metadata，TTS 時讀取並做 time-stretching。
```

---

### 4.6 IF Control（100h）

**特色：**
- A 對 B 下達帶速度指令的 IF 任務，B 照做
- 每輪指令都明確指定速度 + 任務內容
- 指令句型**由 LLM 生成，不要每次一樣**

**對話結構範例：**
```
A：請用快一點的速度說「今天天氣很好」
B：（快速）今天天氣很好。
A：接著，請用很慢的速度從一數到五
B：（慢速）一、二、三、四、五。
A：好，現在用正常速度唸出以下這句話：台灣的珍珠奶茶聞名世界
B：（正常）台灣的珍珠奶茶聞名世界。
A：再用比較快的速度，說出三種台灣小吃
B：（快速）鹹酥雞、蚵仔煎、滷肉飯。
```

**LLM 生成的速度指令句型（不限以下，要多樣化）：**

```
快速類：
  「請用快一點的速度說⋯⋯」
  「可以說快一點嗎，然後⋯⋯」
  「加快速度，唸出⋯⋯」
  「用很快的語速說⋯⋯」

慢速類：
  「請用很慢的速度說⋯⋯」
  「慢慢說，⋯⋯」
  「放慢速度，唸出⋯⋯」
  「你能不能說慢一點，然後⋯⋯」

恢復正常類：
  「好，回到正常速度，說⋯⋯」
  「現在用一般的語速⋯⋯」
  「不用那麼慢了，⋯⋯」
```

**速度狀態追蹤：**
同 4.4 FSM，但狀態切換由 A 的指令句決定（由 LLM 在 dialogue 中生成），
程式讀取每輪的速度指令，標記 metadata，TTS 時套用 time-stretch。

---

## 五、程式架構

```
TASTE2-TW-QWEN2.5-Data/
├── PLANNING.md
├── README.md
├── .gitignore
│
├── conf/                          ← YAML 設定
│   ├── base.yaml                  ← 共用設定（vLLM endpoint、TTS paths、topics）
│   ├── user_agent.yaml
│   ├── daily_conv.yaml
│   ├── if_data.yaml
│   ├── speed_control.yaml
│   └── if_control.yaml
│
├── vllm_server/
│   ├── launch_vllm.sh             ← 啟動 vLLM（單機）
│   ├── launch_vllm.job            ← SLURM job 啟動 vLLM
│   └── check_vllm.py              ← 健康檢查
│
├── generate/
│   ├── base_generator.py          ← 共用 LLM client（vLLM OpenAI-compatible）
│   ├── gen_user_agent.py          ← Type 1
│   ├── gen_daily_conv.py          ← Type 2
│   ├── gen_if_data.py             ← Type 3
│   ├── gen_speed_control.py       ← Type 4（含 FSM）
│   └── gen_if_control.py          ← Type 5
│
├── tts/
│   ├── tts_runner.py              ← 統一 TTS 介面（IndexTTS-2 / BreezyVoice 切換）
│   └── speed_stretch.py           ← audiostretchy time-stretching 工具
│
├── to_parquet/
│   └── to_parquet_v5.py           ← 統一轉 Parquet（相容 TASTE2 SFT loader）
│
└── slurm/
    ├── launch_vllm.job            ← 申請 GPU 跑 vLLM server
    ├── gen_text_array.job         ← 文字生成 array job（多 topic 平行）
    └── tts_array.job              ← TTS 合成 array job
```

---

## 六、vLLM 平行化設計

```
SLURM 申請 32–64 張 H100
  │
  ├── 1–2 張：vLLM server（Qwen2.5-7B，1 張足夠）
  └── 31–63 張：TTS worker（IndexTTS-2 / BreezyVoice）
      每張 GPU 跑 1 個 TTS worker，讀取文字生成結果並合成音訊

文字生成（generate/）：
  - 不需要 GPU，在 CPU 上並行呼叫 vLLM HTTP API
  - 多個 process 同時送 request，vLLM 自動 batching

TTS 合成：
  - 每個 topic 的 TTS 由一個 GPU worker 處理
  - IndexTTS-2 / BreezyVoice 根據 scenario index 決定（前半 / 後半）
```

---

## 七、實作順序

1. **[ ]** 拉 Qwen2.5-7B-Instruct 模型（`huggingface-cli download`）
2. **[ ]** 寫 vLLM 啟動腳本 + SLURM job
3. **[ ]** 寫 `base_generator.py`（vLLM client 封裝 + 重試邏輯）
4. **[ ]** 設計 YAML config（`base.yaml` + 5 個子 config）
5. **[ ]** 實作 Type 1（User/Agent）—— 確認整條 pipeline 可跑通
6. **[ ]** 實作 Type 2（Daily Conversation）
7. **[ ]** 實作 Type 3（IF Data）
8. **[ ]** 實作 Speed FSM + `speed_stretch.py`（Type 4）
9. **[ ]** 實作 Type 5（IF Control）
10. **[ ]** 統一 `to_parquet_v5.py`
11. **[ ]** SLURM array job 大規模生產
