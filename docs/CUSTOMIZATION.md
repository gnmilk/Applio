# Applio 3.6.5 客製化維護背景

記錄日期：2026-10-10。本檔整理此 session 已確認的 repo 狀態，供後續修改使用；agent 工作規則見 [AGENTS.md](../AGENTS.md)。歷史驗證不能代替新修改的驗證。

## 專案與使用情境

- 官方來源：[IAHispano/Applio](https://github.com/IAHispano/Applio)。目前是保有本機環境與資料的 3.6.5 客製化安裝，Git 分支名稱可能變動，先讀目前狀態。
- 新版安裝：`E:\AiVoice\Applio_3.6.5`。舊版候選：`E:\AiVoice\Applio_3.6.4`、`E:\AiVoice\Applio_3.6.4_broken`，存在與否需重新確認。
- 使用者以 Gradio WebUI 操作。Windows 10、RTX 4080 SUPER 16GB、AVerMedia AM350；曾使用 sounddevice／PortAudio、VB-CABLE、Voicemeeter Banana、REAPER、Discord。
- 主要模型情境：RVC v2、40k、HiFi-GAN、RMVPE、Custom Chinese HuBERT（Hugging Face 模型目錄格式）。
- 品質優先於理論最低延遲：自然台灣年輕女性中文聲線、發音穩定、無異常音高／口音、無破音／金屬感、Realtime 穩定，再考量延遲。

使用者已用過的參考值，**不是要求強制覆寫預設值或設定檔**：

| 項目 | 參考值 |
|---|---|
| Pitch／Index Rate／Protect | +9／0.75／0.50 |
| Volume Envelope | 0.63 |
| Chunk／Crossfade／Extra Conversion | 250 ms／0.05 s／2.5 s |
| VAD／Phase Vocoder | ON／ON |
| F0／Embedder | RMVPE／Custom Chinese HuBERT |

250 ms Chunk 曾在遊戲中實測可用；更短 Chunk 曾劣化音質。不要自動調小或大改推論流程。

## 原始碼導覽

| 位置 | 職責／修改時注意 |
|---|---|
| `app.py`、`core.py` | 應用入口與共用操作；執行／import 前檢查副作用 |
| `tabs/realtime/realtime.py` | Gradio 控制、作者原有設定存取、Start／Stop／Refresh、server／client UI |
| `rvc/realtime/core.py` | Realtime 推論、RMS／VAD／靜音、Crossfade 與音量處理 |
| `rvc/realtime/audio.py` | PortAudio 裝置識別、backend refresh、Stream 建立／清理、callback Queue |
| `rvc/realtime/worker.py`、`client.py`、`pipeline.py` | worker／client 配置與模型推論、Embedder 重載流程 |
| `rvc/lib/utils.py` | `load_embedding(...)` 與 `HubertModelWithFinalProj`，多功能共用 |
| `tabs/ui_settings.py` | 新增的 UI 偏好儲存、欄位驗證、頁面 reset、相依 choices／visibility |
| `tabs/inference`、`train`、`tts`、`voice_blender`、`download`、`extra` | 各頁面明確 opt-in 設定記憶 |
| `tests/` | 設定儲存與真實 Gradio 頁面事件的隔離測試、安全檢查入口 |
| `tools/realtime_migration/` | 版本移植證據、歷史備份、回歸測試與診斷腳本；不是每個腳本都適合自動執行 |

## 已提交的修改

以下 hash 是本次工作的追溯點，不代表其他分支／未來工作樹一定仍是相同內容。使用 `git show <hash> -- <file>` 查閱，不要 checkout 舊檔覆蓋現況。

| Commit | 範圍 |
|---|---|
| `0b9da64c` | 移除輸出端重複 RMS 衰減 |
| `1f7fae3f` | 穩定裝置解析、Stream／Start／Stop lifecycle 與 callback 安全 |
| `2cd2b4a2` | Realtime 的 Custom Embedder 不可用時明確失敗 |
| `f5988d7f` | 保存移植證據與驗證工具 |
| `23a3d31c` | UI 設定儲存核心與各頁 reset |
| `1a08389b` | 一般頁面的設定記憶整合 |
| `550754b4` | Realtime 設定記憶與 reset 保護 |
| `f3ef90be` | Download 取樣率連動與安全驗證入口 |

### Realtime 音量

官方舊流程把已生成的模型音訊乘上輸入 RMS，造成額外衰減。已確認本機 3.6.4 曾移除這一乘法，且 3.6.5 仍需要移植：

```python
# 修正前
audio_out: torch.Tensor = self.resample_out(audio_model * vol_t)
# 修正後
audio_out: torch.Tensor = self.resample_out(audio_model)
```

其他 RMS 用途、VAD、Silence Threshold、Warmup、Crossfade、Volume Envelope 保留。靜音 block 清除 Crossfade 殘留。未完成真實模型輸出的 clipping／音量／聽感驗證。

### Windows 音訊裝置與 lifecycle

舊版實際有 name + Host API 識別、啟動前 refresh／重新解析、channels／sample rate 驗證、Stream 失敗清理等客製化；已依 3.6.5 的 block-frame 結構適配。

- `normalize_audio_device_name(...)`、`resolve_audio_device(...)`、`reinitialize_audio_backend(...)` 等主要 backend helper 位於 `rvc/realtime/audio.py`；UI 另有受 Start／Stop 狀態保護的 refresh wrapper。
- 穩定 label 傳到真正開 Stream 的位置，refresh 後重新解析 index。舊格式可相容，模糊／失效裝置不可猜測。
- 同一端點的 fallback 範圍受到限制；MME 短名需來源與目標唯一，重試時重新驗證。
- 未自動 fallback 到 WDM-KS。這是藍畫面後的保守範圍選擇，**不是已證明的藍畫面原因或修復**。
- Start generation 與控制鎖防止 Stop 和延後啟動互相覆蓋；Stream 先清理，再釋放 worker 資源。Callback 不取控制鎖。

Queue／一般 Stop 在舊版與官方 3.6.4 相同，沒有歷史客製化證據。後來透過獨立測試重現空 Queue 阻塞 callback，才做最小非阻塞修正；保留原有 oldest-block／過期 backlog 丟棄策略。不要把這段說成舊版 Queue 移植。

### Chinese HuBERT 與 Cache

- `HubertModelWithFinalProj.from_pretrained(...)` 及本機 Hugging Face 目錄相容性屬原有功能，未任意改寫模型架構。
- 共用 `load_embedding(..., allow_fallback=True)` 保留原有非 Realtime fallback；Realtime 路徑使用 strict 模式，Custom 目錄不可用時禁止偷偷換 ContentVec。
- 官方 worker／client 已比較 selector 與 custom path；測試證明切換 A → B 重新載入、相同路徑重用。不新增重複 Cache 修正。

### WebUI 設定記憶

使用者要求重開後保留數值，並能快速恢復本頁預設值，且不影響作者已自動記憶的欄位。

- 新檔：根目錄 `ui_settings.json`，結構是 `version: 1` 與 `pages` 字典。已加入 `.gitignore`，不要提交使用者偏好或暫存檔。
- 儲存發生於 UI change 事件；單純載入／建立頁面不建立設定檔。以鎖保護讀取合併寫入，暫存檔 + fsync + replace 避免半份資料；損壞／寫入失敗保留原檔。
- 先擷取程式原始預設值再套用已存值，才能正確 reset。無效值使用可用預設值，不直接重寫損壞資料。
- 單檔／批次推論參數各自記憶；TTS 不存原始語句／上傳物件；Training 不存清除／關機／同意條款等危險操作。
- Realtime server 與 client 使用不同 page key；device、model、index 留給作者原本的儲存機制。
- 每頁「恢復本頁預設值」只更新註冊 key；保留其他頁、未知 key 與作者原有記憶欄位。Realtime 執行／啟動中拒絕 reset。
- Download 的 `update_sample_rate_dropdown(model, current_rate=None)` 保留支援的目前 rate，避免 restore／reset 後的 model-change cascade 把 40k 換回 32k。
- Settings、Plugins、Report 等未為了此功能全面改寫；新增需求先確認是否已由作者儲存。

## 證據與後續版本移植

優先查看 `tools/realtime_migration/plan.json` 與 `evidence/`：

- `source-audit.json`、`old-vs-official.patch`：本機舊版相對官方 3.6.4。
- `upstream-3.6.4-to-3.6.5.patch`：官方版本變更。
- `new-preexisting-vs-tag.patch`：本機新版起始差異。
- `final-port.patch`、`final-verification.json`、`regression-tests.log`：本次移植及歷史驗證。
- `windows-bugcheck.json` 與 interrupted 記錄：硬體測試中斷背景。

移植時區分 `Confirmed Custom Modification`、`Probable Custom Modification`、`Unverified Suggestion`、`Already Fixed Upstream`、`Not Applicable`。缺乏三方證據的差異不得全部標為客製化。

## 安全測試與手動驗證

在 repo 根目錄執行：

```powershell
.\env\python.exe -B -X utf8 tests\run_safe_checks.py
```

2026-10-10 最後一次完整執行：117 個 Python 檔案在記憶體編譯、38 項測試 PASS、`assets/config.json` 保持不變。測試涉及 storage／reset、實際 Gradio Python event bindings、模擬音訊 backend／lifecycle／volume／embedder，以及 split-audio 回歸。

**NOT TESTED**：瀏覽器完整操作、實體音訊、GPU inference、Chinese HuBERT 實際音質與 Discord 輸出。先前 GPU／音訊驗證遇到 `WDF_VIOLATION` 後停止，原因未確定。未獲使用者明確同意前，不自動重跑硬體診斷。

使用者手動驗證新功能時：

1. 在一般頁面修改數值，確認「設定已儲存」，重開 app 後比對值與相關選項顯示。
2. 記下作者原有的 Realtime device／model／index，按本頁 reset；確認這些欄位及其他頁設定未改。
3. 在 Download 選 40k，重開及 reset 後等事件完成，再確認 rate 未被連動換成 32k。
4. 若另行恢復硬體測試，先核對 AM350／Voicemeeter 的名稱與 Host API、Chinese HuBERT 目錄及模型／index，再測 Start → Stop → Start；不要由 agent 改 Windows 設定。
5. 音量、靜音殘留、clipping、音質及 Discord 接收端實聽需另行確認；出現新當機停止硬體測試並保留診斷。

## 減少 agent 核准次數

根目錄 `AGENTS.md` 已明確要求需求範圍內自主執行、沿用既有授權、直接使用 Git CLI，以及不要逐個命令另加聊天確認。

平台核准仍由 sandbox／permissions 控制。依 [OpenAI 官方 Sandbox 文件](https://learn.chatgpt.com/docs/sandboxing)，若目前介面提供該選項，可在輸入框下方的權限控制選 **Approve for me**，讓符合條件的請求交給自動審查；這不會取消工作區邊界，審查仍可能拒絕。

對自行管理的 Codex CLI／IDE 設定，官方文件提供以下組合；這是參考範例，**本次未修改任何 Codex 設定**，managed policy 也可能限制可用值：

```toml
sandbox_mode = "workspace-write"
approval_policy = "on-request"
approvals_reviewer = "auto_review"
```

也可在正式核准提示提供時，保存限定的 Git add／commit 或既有安全測試命令規則。不要核准任意 Python／PowerShell 執行或全面解除檔案邊界來省提示。`approval_policy = "never"` 不會給予 sandbox 外的權限；受阻動作可能直接失敗，不能保證更容易完成本專案任務。

後續重要行為、測試入口或限制改變時，同步更新本檔與 `AGENTS.md`；不要保留已過時的「驗證成功」敘述。
