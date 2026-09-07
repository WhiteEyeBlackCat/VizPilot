# VizPilot 開發進度記錄

每個 stage 依序：design critique → implementation → code review → 獨立 verifier 實測 → PASS 後 push。

## Stage 計畫

| Stage | 內容 | 狀態 |
|---|---|---|
| 1 | 後端基礎：FastAPI 骨架、上傳 CSV/XLSX/Parquet、Parquet 快取 + dataset_id、preview/schema API | ✅ PASS |
| 2 | Data Profiler：語意 dtype 推斷、統計、相關矩陣 → DatasetProfile | 進行中 |
| 3 | ChartSpec 模型 + 驗證矩陣 + 規則推薦引擎（含 base score） | 待開始 |
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
