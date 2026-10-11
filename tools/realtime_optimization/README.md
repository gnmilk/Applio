# Realtime 效能與音質量測工具

所有指令在 `E:\AiVoice\Applio_3.6.5` 根目錄使用既有 `env\python.exe`。不安裝套件、不啟動 app、不開音訊 Stream、不改模型／Index／設定／既有 WAV。

## 現在可安全執行的指令

```powershell
.\env\python.exe -B -X utf8 tools\realtime_optimization\benchmark.py --mode audit
.\env\python.exe -B -X utf8 tools\realtime_optimization\benchmark.py --mode self-test
.\env\python.exe -B -X utf8 tools\realtime_optimization\quality.py --input "assets/audios/測試用原音音檔一般聊天.wav"
.\env\python.exe -B -X utf8 tests\run_safe_checks.py
```

audit 只讀 Git、套件 metadata、source/config hash、local files 存在／大小；不 import 模型、不查 PortAudio/CUDA。self-test 用 CPU 合成音檢查 chunk 串接、FLOAT WAV 與報告，不產生虛假的 RVC latency。

每次工具使用 `tools/realtime_optimization/runs/<UTC時間>-<隨機ID>/` 新目錄，拒絕重用既有目錄，沒有 `--overwrite`。runs 與 `*.local.json` 不進 Git。JSON 中文以 UTF-8，CSV 用 UTF-8 BOM，Markdown 用繁體中文。失敗 run 留下原因；缺本機資產不下載。

## 固定模型與參數

`preset.example.json` 是目前 UI 原始預設＋歷史模型情境的範例，不是自動讀取正在執行的 UI。先核對 model/index、Custom Embedder 路徑、pitch、index rate、protect、envelope、VAD、threshold、chunk/crossfade/context 等全部參數。複製為新 `preset.local.json` 後修改；不得覆寫使用者既有 local preset。

輸入 mono 或多聲道均可：多聲道先平均；非 48k 使用 soxr HQ 離線轉為 48k，不計入 inference timer，不正規化音量。chunk 為 48k samples；250 ms 為 12000。crossfade/context 使用秒。第一階段只支援 Custom Embedder、RMVPE、RVC v2，拒絕錄音與裝置選項；Pedalboard 設定尚不支援，開啟時明確報錯。clean_audio 可做單獨 Realtime case，Offline 同條件 A/B 則需關閉。

## 已授權的 GPU Benchmark（不開 Stream）

使用者於 2026-10-10 明確允許恢復 GPU Benchmark，仍禁止實體 Stream。已執行下列正常／profile／Offline 路徑及短段 trace，結果見 [baseline](../../docs/realtime_optimization/baseline_report.md)。`--allow-inference` 是明確 opt-in；不開音效裝置，不代表先前 WDF_VIOLATION 原因已排除。

```powershell
# 正常服務時間基準；不啟用 profile
.\env\python.exe -B -X utf8 tools\realtime_optimization\benchmark.py --mode run --allow-inference --config tools\realtime_optimization\preset.example.json --input "assets/audios/測試用原音音檔一般聊天.wav" --repeats 5 --warmup-chunks 5

# 同條件另跑診斷與原版 Offline reference；正常數據與診斷 pass 分開
.\env\python.exe -B -X utf8 tools\realtime_optimization\benchmark.py --mode run --allow-inference --config tools\realtime_optimization\preset.example.json --input "assets/audios/測試用原音音檔一般聊天.wav" --repeats 5 --warmup-chunks 5 --profile --offline-reference

# trace 在新建的 8 秒 fixture 上執行；原始 WAV 未改寫
.\env\python.exe -B -X utf8 tools\realtime_optimization\benchmark.py --mode run --allow-inference --input tools\realtime_optimization\runs\20261010T084307Z-7c2157b93aff\short_input.wav --repeats 1 --warmup-chunks 5 --profile --trace
```

`--profile` 預設關閉。其 pass 不混入 chunks.csv、deadline、RTF、CPU 或正常 GPU peak。trace 包含 shape/memory 診斷，可能很大且慢，先短樣本後長句。wrapper 只存在於獨立工具的 context，不能在正在執行的 UI 安裝 instrumentation。

本次 8 秒 trace 約 258 MB；留在 ignored run，不自動提交。模型載入前先寫 RUNNING report 與完整 CLI options；中斷後 RUNNING 不能視為 PASS。報告內 `environment` 是無 CUDA 的安全稽核，GPU 實測看 `runtime`。不同背景 GPU 負載要分案記錄，不把正常重跑差異當加速收益。

初始化先跑 `startup warmup + --warmup-chunks`，每輪重設 streaming buffer、VAD 與 seed。每輪真實 startup 仍輸出 zeros（此 preset 前 13 blocks），從 steady-state latency 排除，但保留在 WAV 與 warmup 統計。短 WAV 若全部在 startup 內，模型效能標 NOT TESTED。比較音質時須使用 startup 後同一內容；句首被暖身抑制不能當作模型中文字辨識結果。

本工具是持續送入 chunk 的飽和服務成本模擬，沒有定時等待 arrival、裝置 driver buffer、worker Queue 或網路。正常計時與現有 worker 相同；speech D2H 等待，silent/warmup CPU zeros 可以提前回傳，後塊可能承擔 GPU backlog。不要將「silent 很快」解讀成沒有推論成本；看獨立 trace。

`--seed` 預設 20261010；repeat i 使用 seed+i。RVC 有隨機 latent，跨版本／裝置不保證 bitwise；profiling pass 用第一輪 seed 以比對 instrumentation 的音訊。正式 A/B 使用同 seed、相同 chunk/context、相同 assets hash 與來源版本。

## 產物與欄位

| 檔案 | 內容 |
|---|---|
| report.json／report.md | 環境、Git/source/config/asset/input hashes、精確參數、實測／未測／失敗原因 |
| chunks.csv | repeat、chunk、wall_ms、cpu_ms、warmup、valid_samples；只有正常 pass |
| stages.csv | 每次階段 host inclusive/exclusive、CUDA stream span；只有 profile pass |
| trace.json | 可在支援 Chrome trace 的 viewer 開啟；只有 `--trace` |
| realtime.wav | 第一輪未改寫 Realtime pipeline 的輸出，48k mono FLOAT，不 clip/normalize，也不刪 startup zeros |
| offline.wav | 同資產的官方 Offline Pipeline reference，原本的濾波、padding、F0、FAISS、RMS／peak 正規化保留；48k FLOAT |
| quality.json／quality.csv／quality.md | WAV 數值旗標、chunk RMS、選用外部 F0／逐字稿比較 |

`performance.latency_ms` 含 p50/p95/p99；deadline 為 block_frame/48000，只計 `wall_ms > deadline`。RTF=正常 service time／有效輸入音訊時間，最後 partial chunk 的 padding 不增加分母。warmup 零輸出不列入 steady-state，但最後 warmup 的 GPU 工作可能延至第一個 steady chunk。`completed_passes` 另提供含最後 CUDA fence 的整輪完成 RTF，包含 startup 與輸出保留／RSS 取樣，不混入 chunk latency。process CPU time 會累計 CPU threads，`cpu_core_equivalents` 可大於 1；不是整台機器 CPU 百分比。RSS 在 chunk 後取樣且包含保留 waveform，非精確峰值；CUDA allocated/reserved peak 含常駐權重／Index，在正常 pass 重設 peak。沒有 CUDA 時欄位為 null。

stage summary 先加總同一 chunk 的多次呼叫（如 SOLA 兩次 conv1d），才算 percentile。inclusive 可重疊，不能相加。CUDA Event 跨度可含 CPU 排程空洞，不能當純 kernel time。未分段操作留在 residual scope，詳見 [架構分析](../../docs/realtime_optimization/architecture.md)。

## Offline／Realtime 與外部評估介面

完成 run 後，用其輸出路徑替換以下範例；不要拿來源不明的舊 `_output.wav` 宣稱同參數 baseline。

```powershell
.\env\python.exe -B -X utf8 tools\realtime_optimization\quality.py --input "assets/audios/測試用原音音檔一般聊天.wav" --offline "tools/realtime_optimization/runs/實際run/offline.wav" --realtime "tools/realtime_optimization/runs/實際run/realtime.wav" --benchmark-report "tools/realtime_optimization/runs/實際run/report.json"
```

範例中的「實際run」須替換為本次生成的目錄。若知道由 SOLA/context 造成的內容延遲，加入 `--realtime-delay-samples N`／`--offline-delay-samples N`；工具用宣告的 delay 對 source 填前導 zeros，不移動輸出或 chunk 邊界。沒有 delay 不判定相對來源的非預期靜音。邊界跳變門檻 0.1、靜音 -60 dB、活躍來源 -40 dB 都只是診斷規則，預設不能當成聽感結論。

外部 F0：CSV 第一行 `time_s,hz`，其後每行時刻與 F0；0 表示 unvoiced。使用同一獨立估計方法、hop、共同有效區段、相同且嚴格遞增時間網格，先自行校正延遲，工具不自動內插 V/UV。指定 `--f0-source`、`--f0-offline`、`--f0-realtime`、`--pitch-shift 9`。回報 pitch-shift 校正後 cents 誤差、共同 voiced 比例、voicing agreement 與 log-F0 上下行趨勢；不能用生成器輸入 pitch cache 充當實際輸出 F0。

已有同 run 的 Offline／Realtime WAV 時，可用既有 RMVPE 獨立估計；不下載新模型、不套用高音修正、不開 Stream：

```powershell
.\env\python.exe -B -X utf8 tools\realtime_optimization\f0_evaluation.py --benchmark-report tools\realtime_optimization\runs\20261010T084344Z-a54afaa6a23c\report.json --allow-inference --realtime-delay-ms 60 --offline-delay-ms 0
```

輸出 source/offline/realtime 的 raw 與 aligned CSV、comparison.csv、f0_report.json／md。預設共同起點是該 report 的 chunks.csv 第一輪實際 warmup 結束後一塊；本 case 為3.5秒。`--start-seconds` 不得早於 startup；report 的 `inference.f0_up_key` 省略時使用pipeline預設0。autotune/proposed_pitch 的動態升調不能使用固定升調誤差，明確拒絕。delay只接受10ms整數倍，不補值／內插 V/UV；固定延遲與±20ms敏感度並列，不以最佳結果挑延遲。60ms是本case context/return尺寸的近似，仍需人工校正音節／SOLA位移。

例如用本次最後一份 F0 估計重跑品質報告：

```powershell
.\env\python.exe -B -X utf8 tools\realtime_optimization\quality.py --input "assets/audios/測試用原音音檔一般聊天.wav" --offline tools\realtime_optimization\runs\20261010T084344Z-a54afaa6a23c\offline.wav --realtime tools\realtime_optimization\runs\20261010T084344Z-a54afaa6a23c\realtime.wav --benchmark-report tools\realtime_optimization\runs\20261010T084344Z-a54afaa6a23c\report.json --realtime-delay-samples 2880 --offline-delay-samples 0 --f0-source tools\realtime_optimization\runs\20261010T090412Z-7c965d45f735\source_aligned.csv --f0-offline tools\realtime_optimization\runs\20261010T090412Z-7c965d45f735\offline_aligned.csv --f0-realtime tools\realtime_optimization\runs\20261010T090412Z-7c965d45f735\realtime_aligned.csv --pitch-shift 9
```

補充情境逐案可重跑相同 benchmark，替換 input 為既有 `(疑問句)`、`(相似音密集測試)`、`(快速話測試)`、`(低音量、氣音、輕聲)`、`(情緒快速切換)`、`(遊戲場景)` WAV。本次每案 `--repeats 1 --warmup-chunks 5 --offline-reference`，無profile。60秒新fixture為 `runs/20261010T085641Z-08ad59ef0c50/long_input_60s.wav`，原始161秒音檔保持原樣；新增fixture來源hash記錄在該目錄preflight.json。每案完整命令與run路徑見baseline的suite evidence；增加重複次數才適合研究尾端分布。

外部逐字稿：JSON 字串 key `source`、`offline`、`realtime`、`provenance`（描述 ASR 名稱／版本／設定或人工逐字）。使用 `--transcripts 外部檔案.json`；不下載 ASR，CER 只做 NFKC、去標點／空白，不偷做繁簡轉換。先對相同有效 speech 範圍逐字；ASR 的誤辨與繁簡差異要人工複查。

## 人工 A/B 與裝置驗證

1. 用同一句一般聊天、疑問句、四聲對比、相似音、氣音／低音量、高低音跳轉、快速遊戲句及 30–60 秒長句。記下詞、字與時間區段；原音、Offline、Realtime 用共同有效區段比較。
2. 等響度試聽時只調播放器的比較增益，不改原始 FLOAT WAV。匿名 A/B，記錄錯字／聲調、台灣口音、年輕自然度、金屬感、邊界 pop/click、氣音與句尾截斷、長句飄移、V/UV 切換。不要將 objective flag 或 CER 當單一品質排名。
3. 另行恢復實體測試時，先核對穩定 Device Name + Host API 與現有路由，不修改 Windows 設定。先 Start→Stop→Start，再安靜／正常／輕聲與長句，最後加遊戲負載。用 OBS／接收端錄音確認 clipping 與重複音節。
4. 真正端到端延遲要用可識別脈衝／loopback 或已校準的共同時鐘 timestamps；記下輸入輸出裝置、Host API、sample rate、driver buffer、路由、Discord/OBS 處理。純 Pipeline 的 p95 不是麥克風到 Discord 接收端的延遲。
5. 若再有 driver 當機或藍畫面，停止硬體測試並保留 run／系統診斷。本次不宣稱已排除原因。
