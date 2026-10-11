# Applio 3.6.5 客製化工作指引

本檔適用於此 repo 的後續 agent。詳細背景、既有修改與測試方式見 [docs/CUSTOMIZATION.md](docs/CUSTOMIZATION.md)。以目前原始碼及 Git 狀態為準，不要把本檔的歷史紀錄當成新的測試結果。

## 工作方式與自主執行

- 使用繁體中文溝通。使用者提出修改需求就是要求實作；完成分析、最小修改、適當驗證及交付，不要只提供建議或停在計畫。
- 需求明確時，直接進行 repo 內的讀取、搜尋、原始碼與測試修改、文件更新、差異檢查及安全測試，不必逐步詢問「可以開始嗎」。
- 一般實作細節、函式命名、測試拆分及低風險設計由 agent 決定。先簡短說明假設並繼續工作；只有會改變需求範圍、品質取捨或安全界線的缺失資訊才需要詢問。
- 使用者授權在同一任務中持續有效。已要求 commit 時，直接規劃並執行合理拆分的本地 commit，不要再以聊天要求確認每筆提交。
- 優先直接使用 Git CLI，不必透過 GitKraken。保留既有分支與使用者未提交修改；不要自動 checkout、stash、reset、amend、rebase 或建立會缺少本機環境／模型的 worktree。
- 依功能拆分 commit，每筆包含相關測試；避免把儲存核心、各頁整合、Realtime 行為及不相關修正全部塞入同一筆。明確列出檔案再 `git add -- ...`，不要無差別 `git add .`。
- 使用者尚未要求提交時，不要為了結束工作而自行提交；可完成所有修改與驗證，交付可直接提交的結果。
- 不自動 push、merge、發版、發布、傳訊息或改遠端設定。這些行為需要使用者明確授權。
- 優先 `rg` 搜尋。可平行讀取互不依賴的資料；相依修改、測試及 Git 寫入依序執行。不要為了省事輸出大量檔案或重複整套驗證。
- 定期簡短回報已確認的結果與下一步；遇到小型版本差異、測試失敗或安全可修復的問題，自行修正並繼續。

### 核准的界線

上述規則減少的是 agent 自行增加的詢問，不會覆寫系統／開發者指令、sandbox、managed permissions、工具的強制核准或自動核准審查。

- 優先使用目前可用的工作區權限與已核准的命令規則。
- 工具要求提升權限時，使用正式核准流程，清楚描述具體動作；不要先在聊天問一次，再另外彈出相同動作的核准。
- 核准遭拒時，不得換工具或改寫命令來繞過；繼續可獨立完成的工作，最後說明被阻擋的具體動作。
- 不擅自修改 Codex 全域設定、權限設定、`.codex` 或 `.agents`，也不把本檔當作取消安全限制的依據。

## 保護範圍

- 唯一修改目標是目前的 Applio 3.6.5 repo，原始安裝位置為 `E:\AiVoice\Applio_3.6.5`。先確認 cwd／repo，不能假定候選路徑一定存在。
- `E:\AiVoice\Applio_3.6.4`、`E:\AiVoice\Applio_3.6.4_broken` 僅供讀取比較。
- 不執行 `run-install.bat`；不重建、覆蓋、刪除 `env`；不安裝或更新 PyTorch、CUDA、Gradio、sounddevice 或其他依賴。
- 不覆蓋、刪除或重設 `logs`、`assets`、模型、index、checkpoint、訓練資料或現有設定檔。必要的設定寫入應由使用者操作既有 UI 或明確授權的功能完成；測試使用暫存設定。
- 不自動修改 Windows Audio、預設音訊裝置、音量、取樣率、驅動、VB-CABLE、Voicemeeter、REAPER 或 Discord 設定。
- 不將舊版完整 Python 檔覆蓋新版。保留 3.6.5 官方功能，採用最小補丁。
- 不還原或格式化使用者既有修改。非 Git 情境先備份本次修改檔；Git 情境保留起始 status/diff，需要時另存補丁或限定檔案備份。

## 必須保留的 Realtime 行為

- `rvc/realtime/core.py` 的輸出是 `self.resample_out(audio_model)`，不能重新加回 `audio_model * vol_t`。輸入 RMS 仍供音量診斷／靜音判斷使用。
- 保留 VAD、Silence Threshold、Warmup、Crossfade、Volume Envelope、Phase Vocoder 及既有 inference pipeline。音量修正不是移除所有音量控制。
- 裝置識別以 Device Name + Host API 為主，啟動及實際開 Stream 前用當下裝置清單解析；不可永久保存或硬編碼 PortAudio index。
- 同名或不可用的端點應明確報錯，不猜測；先驗證 channels、sample rate、float32 與 backend 參數。
- Stream 存在或 Start 尚未完成時不得重新初始化 PortAudio；保留 start generation、控制鎖與失敗啟動的清理順序。
- 音訊 callback 不得加入磁碟 I/O、大量 logging、昂貴運算、等待、控制鎖或阻塞 Queue 讀取。保留目前音訊連續性與過期 backlog 策略，變更前先證明必要性。
- Realtime 選 Custom Embedder 時，目錄不可用必須報錯，不得默默換 ContentVec。共用載入器的非 Realtime 預設行為不要連帶改掉。
- Embedder 快取必須同時辨識 `embedder_model` 和 `embedder_model_custom`；目前官方流程已具備此判斷，不要重複新增快取層。
- 優先相容 RVC v2／40k／HiFi-GAN／RMVPE／Custom Chinese HuBERT；不要擅自換架構、F0 或 Embedder。

## WebUI 設定記憶與恢復預設值

- `tabs/ui_settings.py` 是新增記憶功能的共用入口。`ui_settings.json` 與作者的 `assets/config.json` 分開，皆不得提交使用者內容。
- 以 `remember_page(...)` 明確註冊欄位。維持值域／型別／路徑驗證、atomic replace、損壞檔保留、頁面間隔離及原始預設值的擷取順序。
- 「恢復本頁預設值」只恢復本功能管理的欄位。作者原有自動記憶欄位、其他頁及未註冊 key 不受影響。
- Realtime 的 input/output/monitor device、model/index 原本由作者儲存，必須排除；server 與 client 偏好分開。
- Realtime 執行中或啟動中不可 reset；client 使用 session State，不能以跨使用者 global bool 代替。
- 相依 dropdown／visibility 需驗證「載入 → 事件連動」及「reset → 事件連動」，避免值被後續 callback 改回。
- 不記憶同意條款、檔案上傳物件、原始 TTS 文字或破壞性訓練動作（例如清除／關機）。新增欄位時逐項判斷，不做全 UI 反射式自動儲存。

## 驗證

Python 使用既有 `env\python.exe`，不要換成系統 Python，也不要因缺少 pytest 安裝依賴。

安全回歸入口（repo 根目錄、PowerShell）：

```powershell
.\env\python.exe -B -X utf8 tests\run_safe_checks.py
```

- 此入口在記憶體編譯原始碼，執行使用隔離 backend 的 UI／Realtime／split-audio 測試，並檢查 `assets/config.json` 未被改動。它不等於真實 Realtime 驗證。
- 歷史基準：2026-10-10，117 個 Python 檔案、38 項測試通過；未來以實際輸出為準。
- 本次 session 曾發生 Windows `WDF_VIOLATION` 藍畫面。除非使用者明確允許恢復硬體驗證，不自動執行實體音訊 Stream、GPU inference、`audio_smoke.py`、`offline_smoke.py` 或任何可能觸發它們的腳本。原因尚未確認，不能宣稱程式碼修正已排除藍畫面。
- 先讀測試腳本及 import side effects，再執行；不要直接啟動 app 或 import 全部頁面來「測試」，部分上游模組 import 會清理 cache／建立目錄。
- 修改需要時增加有意義的 regression tests；純文件／空白調整做對應檢查即可，不必跑硬體測試或反覆重跑整套。
- 本 repo 部分既有 tracked 檔含 CRLF；差異檢查可用下列命令，不要為修正檢查輸出而整檔改行尾：

```powershell
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check
```

## 交付

明確說明修改檔案、行為、原因與實際驗證結果；區分 `PASS`、`FAIL`、`NOT TESTED`。不能用靜態／mock 測試通過宣稱麥克風、GPU、Chinese HuBERT 音質或 Discord 實際語音已驗證。需要手動驗證時提供具體操作步驟。若已提交，列出 commit hash 與範圍；沒有 push 就明說。
