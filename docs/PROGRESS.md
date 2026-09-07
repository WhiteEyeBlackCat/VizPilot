# VizPilot 開發進度記錄

每個 stage 依序：design critique → implementation → code review → 獨立 verifier 實測 → PASS 後 push。

## Stage 計畫

| Stage | 內容 | 狀態 |
|---|---|---|
| 1 | 後端基礎：FastAPI 骨架、上傳 CSV/XLSX/Parquet、Parquet 快取 + dataset_id、preview/schema API | ✅ PASS |
| 2 | Data Profiler：語意 dtype 推斷、統計、相關矩陣 → DatasetProfile | ✅ PASS |
| 3 | ChartSpec 模型 + 驗證矩陣 + 規則推薦引擎（含 base score） | 進行中 |
| 4 | Chart render API：後端聚合/抽樣/heatmap 矩陣 | 待開始 |
| 5 | LLM 層：LLMProvider 抽象、OpenAI-compatible 本地 provider、合併排序、fallback | 待開始 |
| 6 | 前端 SPA（React+TS+Vite+Tailwind+react-plotly）＋整合驗證 | 待開始 |

## Stage 記錄

### Stage 1 — 後端基礎與上傳（PASS，第 1 次驗證即通過）

- **修改內容**：`backend/` 全新建立 — FastAPI app factory + CORS（僅 localhost）、`POST/GET /api/datasets`、`GET /{id}`、`/{id}/preview`、`/api/health`；loader（CSV utf8-lossy fallback / xlsx 第一 sheet / parquet）；DatasetStore（parquet+json 落盤、in-memory cache、lazy reload、原子 metadata 寫入）；`serialization.py` 確立全專案 JSON 慣例（NaN/±inf→null、datetime→ISO）。
- **測試結果**：pytest 33 passed / 0 failed。
- **Review**：design critique 1 BLOCKING（NaN 序列化）+ 8 建議全部落實；code review 無 BLOCKING，3 個 SUGGESTION（原子寫入、註解修正、排序測試）已修。
- **Verifier 結果**：PASS — 獨立實測 pytest、uvicorn + curl 全流程（含 path traversal、210MB 大檔 413、xlsx/parquet 上傳、raw JSON 無 NaN/Infinity）。
- **尚存風險**：(1) 超大上傳的 multipart body 會先被 framework spool 到磁碟才回 413（不佔記憶體，已註明；嚴格早退需 middleware）。(2) in-memory cache 無上限（刻意接受，單人短 session）。(3) 巢狀型別（List[Float] 含 NaN）會 500 大聲失敗，屬 D9 範圍外。

### Stage 2 — Data Profiler（PASS，第 1 次驗證即通過）

- **修改內容**：`backend/app/profiling/`（models/types/profiler）— 語意型別推斷（boolean/datetime/numeric/categorical/text/id/unknown，含字串日期多格式、千分位/貨幣/百分比清洗、camelCase id、低基數整數→categorical）；`cast_params` 存進 profile、`apply_semantic_casts` 純重放（Stage 4 契約）；統計＋逐 pair Pearson（pairwise 排除 null/NaN、>30 欄截斷）；`GET /api/datasets/{id}/profile`＋磁碟/記憶體雙 cache 帶 `profile_version` 版本戳。
- **測試結果**：pytest 101 passed（新增 68，Stage 1 全保留）。
- **Review**：design critique 3 BLOCKING（cast 參數入 profile、cache 版本戳、id 先於 text）+ 10 建議全落實；code review 1 BLOCKING（long-format 重複日期把 daily 誤判 irregular，diff 前先 unique()）已修＋補測試，另修字串數字 id 與 Int64 路徑不一致。
- **Verifier 結果**：PASS — 髒資料 CSV 逐欄型別與統計值獨立對答案全部吻合（含 %d/%m/%Y 方向驗證）、long-format daily、correlation 數值/對稱、cache 命中與版本失效重算、round-trip 重放、raw JSON 零 NaN/Infinity。
- **尚存風險**：(1) `datetime_format="infer"` 的欄重放時仍由 polars 推格式（弱決定性，實測穩定）。(2) numeric 欄含 NaN 時 unique_count 差 1。(3) 歐式小數（€1.000）誤判，屬明確不做範圍。
