# Realtime 第一階段 Baseline 報告

日期：2026-10-10。**使用者已允許恢復 GPU Benchmark；CUDA 純 Pipeline、Profiling、Offline 對照及獨立輸出 F0 已實測，沒有開實體音訊 Stream。** 本次維持 RVC 架構與 production 推論程式，不安裝依賴、不改既有模型、Index、音檔或設定。成功量測不能證明先前 WDF_VIOLATION 原因已排除。

## 版本、工作樹與參數

| 項目 | 本次確認 |
|---|---|
| repo／Applio | `E:\AiVoice\Applio_3.6.5`／assets version 3.6.5 |
| branch／HEAD | `v3.6.5/self-enhance-realtime-perfermance`／`f3ef90be7dde36a0cdf909986cfc2d9dbe62d15b`，保持不變 |
| 起始 status／隔離 | 未追蹤 AGENTS.md、docs/CUSTOMIZATION.md；新增獨立工具、測試及文件，不切分支／worktree／stash／reset |
| OS／Python | Windows 10 build 19045／既有 env Python 3.12.15 |
| PyTorch／torchaudio | 2.11.0+cu128／2.11.0+cu128；CUDA build 12.8 |
| GPU／driver | 實測 RTX 4080 SUPER、16376 MiB／591.86；device cuda:0 |
| 模型 | TaiwanFemale3_CNHubert_v3_90e_28890s.pth；RVC v2／40k／HiFi-GAN |
| Embedder／F0／Index | Custom Chinese HuBERT／RMVPE／10,000 vectors，Torch GPU 全搜尋，未切 CPU fallback |
| precision／threads | FP32／torch intra-op threads 24 |
| chunk／crossfade／extra | 250 ms（12000@48k）／50 ms／2.5 s；context 2.81 s |
| 控制參數 | pitch +9、index 0.75、protect 0.5、volume envelope 0.63、Phase Vocoder 開啟 |
| 靜音／後處理 | VAD mode 3／30 ms、threshold −60 dB；clean_audio/post_process/autotune/proposed_pitch 關閉 |
| seed／warmup | seed 20261010；repeat i 用 seed+i；初始化 13+5 chunks，各輪實際 startup 13 chunks 另計 |

範例 preset 是本次量測條件，不等於已核對正在執行的 UI。精確參數、模型／Index／Embedder／RMVPE SHA256、來源程式及設定 hash 見 [gpu_baseline.json](evidence/gpu_baseline.json)。原本 RMS 衰減修正、裝置解析、lifecycle、非阻塞 Queue、Custom Embedder 嚴格載入與 UI 記憶保持原樣。

**背景負載限制：**開始前 nvidia-smi 顯示 8013 MiB／47% 使用率；量測途中一個快照為 8186 MiB／91%；工具完成後一個快照仍為 7383 MiB／69%。沒有停止使用者程序，也未確認負載是哪個應用。這些是離散快照，不是連續記錄，不能從中推定平均 GPU 使用率或遊戲效能。正常單輪重跑 p95 降至 167.45 ms，顯示負載／排程有明顯變動；本報告不承諾 idle GPU 的加速百分比。

## 實際正常效能

固定原音：`assets/audios/測試用原音音檔一般聊天.wav`，48k mono、983040 samples／20.48 s。SHA256 `123ee84ba56e523653289f8ba8e80b588d4303eaef5c7f67356274aff7d46a03`。

主要 run：`20261010T084344Z-a54afaa6a23c`，五輪正常 pass；之後才執行獨立診斷與 Offline，不將其同步／trace／Offline 開銷混入下表。

| 指標 | 實測 |
|---|---|
| 穩態／startup chunks | 345／65；每輪 69／13 |
| chunk wall p50／p95／p99 | **138.236／200.858／222.157 ms** |
| deadline／miss | 250 ms／**2/345（0.580%）**（254.49、253.10 ms） |
| 穩態服務時間 RTF | **0.5273** |
| process CPU p50／p95／p99 | 125／203.125／218.75 ms；Windows process timer 解析度有限 |
| CPU core equivalents | 0.963；不是整台機器 CPU 百分比 |
| 每輪整體完成 RTF（含尾端 CUDA fence） | 0.4944、0.4509、0.4498、0.4594、0.5978 |
| RSS chunk 後觀測峰值 | 2385551360 bytes，約 2275.04 MiB；包含輸出保留，非精確瞬時 peak |
| Torch CUDA allocated／reserved peak | 986926592／1092616192 bytes，約 941.21／1042 MiB；含常駐模型／Index，不等於 nvidia-smi 全機占用 |
| End-to-End 裝置 latency | **NOT TESTED** |

RTF 是 worker 相同 process_audio 的飽和服務成本模擬；未加入定時 arrival、worker IPC、callback、driver buffer、路由與 Discord/OBS。speech D2H 本來會等待；silent/startup 可以提前返回，其 GPU backlog 可能延至下一塊。0.58% 服務 deadline miss 不能直接等同實體 underrun。p99 僅代表本次樣本分布，不是長期穩定性保證。

8 秒預檢 run `20261010T084313Z-4321aee030c1`：19 穩態 chunks、p95 138.67 ms、0 misses。正常同 seed 單輪重跑 `20261010T084559Z-b78ee0e39ad3`：69 chunks、p95 167.45 ms、0 misses、RTF 0.3913。未把不同負載 run 合併掩蓋差異。

## 階段診斷：與正常 latency 分開

以下為主要 run 的診斷 pass，69 穩態 chunks；輸出相關 scope 只在62個非靜音 chunks觀測。CUDA Event 是當前 stream 跨度，含 enqueue 間隙與等待，不是純 kernel time。scope inclusive 有重疊，不能相加。

| scope | host p50／p95／p99 ms | CUDA span p50／p95／p99 ms |
|---|---|---|
| Generator | 96.896／118.999／121.765 | 76.085／99.006／104.570 |
| RMVPE／F0 | 15.525／18.531／20.140 | 60.718／73.253／77.074 |
| HuBERT | 7.739／9.566／10.121 | 17.096／20.529／21.279 |
| SOLA correlation（兩次合計） | 0.237／0.435／0.627 | 14.854／18.248／19.388 |
| Phase Vocoder | 4.893／5.691／7.140 | 5.959／6.695／8.216 |
| Index retrieval | 0.481／0.814／1.178 | 1.593／2.954／3.972 |
| Volume Envelope | 0.402／0.924／1.019 | 1.323／3.028／5.609 |
| buffer write（同 chunk 加總） | 0.146／0.258／0.707 | 0.867／1.466／1.725 |
| input resample | 0.260／0.483／1.084 | 0.012／1.463／1.550 |
| output resample | 0.178／0.334／0.411 | 0.013／1.265／1.381 |
| VAD CPU | 0.071／0.131／0.257 | 0.094／0.194／0.332 |

frontend/conversion/chunk residual、整體 scope 及完整欄位保留於 JSON／stages.csv。noise reduction/post_processing 未啟用，不填虛構零值。

短段 trace run `20261010T084929Z-3aef61739195`：32 chunks，排除13 startup後19穩態；trace約258 MB，包含實際 kernel／memcpy／CUDA runtime Events。透過 External id 連回 CPU op，歸屬最內層 scope，得到另一個角度：

| scope | 純 kernel p50 ms | 平均 kernels／chunk |
|---|---|---|
| HuBERT | **22.664** | 248 |
| SOLA correlation | **13.989** | 8（18個非靜音 chunks） |
| Generator | 4.807 | **1043** |
| RMVPE／F0 | 3.494 | **832** |
| Index retrieval | 0.127 | 21 |
| Phase Vocoder | 0.0905 | 41（18個非靜音 chunks） |

kernel 及 memcpy sum 不等於 wall time；並行 stream、profiler 開銷、不同時段背景負載均會影響解讀。短段診斷有166次 cudaStreamSynchronize，host等待總計1679.5 ms；20次 cudaDeviceSynchronize包含工具每塊同步，不能宣稱全部是 production 問題。資料見 [gpu_trace_summary.json](evidence/gpu_trace_summary.json)。

**成本排序須區分觀點：**可見 host 模組呼叫前三為 Generator、RMVPE、HuBERT；純 GPU kernels 前三為 HuBERT、SOLA、Generator。Generator/F0 的啟動與同步成本值得先定位；不能因 host scope 大就宣稱 Generator 網路計算最慢。50 ms Phase Vocoder 與目前 10k Index 不是首要瓶頸。

## GPU 重現性與 instrumentation

正常同 seed 重跑與主要 run 第一輪，最大 sample 差異0.000556、RMS差異0.00001843。主要 profiling 對正常第一輪最大差異0.000472；短段 trace 對其正常 pass 0.000518。小幅差異在無 instrumentation 的 GPU 重跑亦存在，尚未釐清具體 nondeterministic kernel；**GPU bitwise 等價仍未證明**。沒有以放寬測試容差宣稱「完全不變」。CPU fixture 的 waveform／SOLA state bitwise 測試通過。

本階段只新增獨立工具，不修改 production 推論；profiling 預設關閉且僅作用於工具建立的獨立模型物件。新增 RUNNING checkpoint，在載入模型前留下 run 及 CLI 參數；異常中斷可識別未完成的執行，非成功數據。

## 音質數值 baseline

同模型／Index／Embedder／控制參數的官方 Offline 與未修改 Realtime WAV 已產生。Offline 原本的 high-pass、padding、F0、CPU FAISS、RMS／peak 正規化保留；其長度20.46秒，Realtime20.48秒。不能將兩條路徑當數值等價，也不能從輸出較大推定較自然。

| 指標 | 原音 | Offline | Realtime |
|---|---|---|---|
| NaN／Inf samples | 0 | 0 | 0 |
| clipping samples（abs≥1） | 0 | 0 | 0 |
| peak | 0.17871 | 0.42794 | 0.38440 |
| 全檔 RMS | 0.02335 | 0.06554 | 0.05002（含startup） |
| 250ms網格最大邊界 sample 跳變 | 0.01694 | 0.03081 | 0.01570 |
| 邊界跳變超過0.1旗標 | 0 | 0 | 0 |

Realtime startup 前3.25秒為既有零輸出，不判為內容遺失或 ASR 錯誤。以 RT 60 ms／Offline 0 ms 的宣告近似 delay及 −40 dB來源活動／−60 dB靜音門檻，兩份輸出皆無額外意外靜音旗標；門檻可能漏掉輕聲，不能宣稱無截字／click／pop。細節見 [gpu_quality.json](evidence/gpu_quality.json)。

獨立 F0：使用既有 RMVPE FP32從來源與兩份完整輸出波形估計，16k、10ms hop、threshold0.03；不是生成器 pitch cache。排除startup，從來源3.5秒起，RT假設60ms延遲、Offline0ms，升調校正+9半音，共同1693 frames：

| 指標 | Offline | Realtime |
|---|---|---|
| 共同 voiced frames | 1039 | 992 |
| V/UV agreement | 96.69% | 93.50% |
| 中位絕對F0誤差 | 6.01 cents | 10.75 cents |
| 共同 voiced 連續frames的上下行趨勢agreement | 85.83% | 84.97% |

RT延遲40/50/60/70/80ms時中位誤差為44.90/27.44/10.75/11.78/28.08cents，顯示對齊敏感。60ms依context/return尺寸事先假設，不以最小誤差挑選延遲；SOLAoffset、VAD凍結context及估計誤差尚未逐音節校正。**這些數字不是中文四聲正確率或女聲自然度分數。** 外部逐字稿／ASR介面保留，未安裝ASR，中文內容一致性與台灣口音仍需人工。資料見 [gpu_f0.json](evidence/gpu_f0.json)。

## 補充情境與長段

補充既有低音量／氣音、快速話、疑問句、相似音、情緒切換、遊戲場景，以及161.088秒原始錄音首60秒的新fixture；每案一輪、相同模型參數、不開profile，另產生Offline參考。單案少量chunk的p99只作描述；不將情境／背景差異當最佳化收益。實際表與產物見下節及 [gpu_suite.json](evidence/gpu_suite.json)。首60秒不是新錄製的連續長句，不能單憑檔名宣稱中文長句聽感穩定。

| 情境 | 穩態chunks | p95 ms | deadline miss | RTF | RT／Offline WAV |
|---|---|---|---|---|---|
| 低音量、氣音、輕聲 | 52 | 72.842 | 0 | 0.2546 | [RT](../../tools/realtime_optimization/runs/20261010T085713Z-a81b13d8895a/realtime.wav)／[Offline](../../tools/realtime_optimization/runs/20261010T085713Z-a81b13d8895a/offline.wav) |
| 快速話測試 | 42 | 197.240 | 0 | 0.3856 | [RT](../../tools/realtime_optimization/runs/20261010T085732Z-63e48f1175aa/realtime.wav)／[Offline](../../tools/realtime_optimization/runs/20261010T085732Z-63e48f1175aa/offline.wav) |
| 疑問句 | 67 | 159.743 | 0 | 0.4925 | [RT](../../tools/realtime_optimization/runs/20261010T085752Z-fa7f3ebaa017/realtime.wav)／[Offline](../../tools/realtime_optimization/runs/20261010T085752Z-fa7f3ebaa017/offline.wav) |
| 相似音密集測試 | 53 | 131.613 | 0 | 0.4533 | [RT](../../tools/realtime_optimization/runs/20261010T085819Z-7d0da6e7c050/realtime.wav)／[Offline](../../tools/realtime_optimization/runs/20261010T085819Z-7d0da6e7c050/offline.wav) |
| 情緒快速切換 | 74 | 74.648 | 0 | 0.2292 | [RT](../../tools/realtime_optimization/runs/20261010T085843Z-dcbbabce7f83/realtime.wav)／[Offline](../../tools/realtime_optimization/runs/20261010T085843Z-dcbbabce7f83/offline.wav) |
| 遊戲場景 | 57 | 139.256 | 0 | 0.4674 | [RT](../../tools/realtime_optimization/runs/20261010T085903Z-ea9db09fadc5/realtime.wav)／[Offline](../../tools/realtime_optimization/runs/20261010T085903Z-ea9db09fadc5/offline.wav) |
| 60秒長段 | 227 | 196.481 | 0 | 0.4398 | [RT](../../tools/realtime_optimization/runs/20261010T085928Z-f4c976569ff5/realtime.wav)／[Offline](../../tools/realtime_optimization/runs/20261010T085928Z-f4c976569ff5/offline.wav) |

所有補充輸出NaN/Inf、clipping及>0.1邊界旗標皆為0，服務deadline miss皆0。未對齊時「情緒切換」8.75–9.00秒及「遊戲場景」3.75–4.00秒各被標低RMS；加上事先宣告60ms延遲後旗標消失，原輸出並非全零，不以此宣稱VAD吞字。仍建議人工複查這兩段。

同參數CPU重播torchaudio resampler／WebRTC VAD與threshold：穩態低音量8/52、快速話2/42、疑問8/67、相似音1/53、情緒8/74、遊戲4/57、長段24/227塊輸出全零，均與CPU重播gate預測一致。這不是直接GPU gate telemetry，不能證明靜音判斷正確；低音量8塊尤其需人工聽是否包含氣音／句尾。詳見 [gpu_vad_diagnostics.json](evidence/gpu_vad_diagnostics.json)。

## 驗證、限制與使用者操作

- **PASS**：GPU短段、主要五輪、獨立Events、短段CUDA kernel trace、Offline reference、獨立RMVPE波形F0及CPU品質診斷；精確run資訊見evidence。
- **PASS**：124個Python檔案記憶體編譯、65項安全測試（含27項新增）、assets/config.json未變；見 [verification.json](evidence/verification.json)。隔離裝置測試只用mock，不等於開Stream。
- **NOT TESTED**：中文ASR／逐字內容一致性、自然年輕女聲／台灣口音、電子感／click的主觀聽感、麥克風／Discord／OBS／遊戲實際穩定性、端到端latency、先前藍畫面原因。
- **未實作Prototype**：背景負載變動且GPU bitwise尚未證明；主要候選涉及launch／同步／模型運算，尚無已證明值得採用且不改推論結果的局部修改。沒有盲目換架構、F0、Embedder、FP16、ONNX或compile。

使用者只需人工A/B試聽：先跳過Realtime前3.25秒，來源／Offline／Realtime以共同內容比較，播放器等響度；記錄疑問句聲調、相似音、氣音／清音、句尾截斷、金屬感及長段飄移。每案WAV位於各run目錄；主要 [Realtime WAV](../../tools/realtime_optimization/runs/20261010T084344Z-a54afaa6a23c/realtime.wav)、[Offline WAV](../../tools/realtime_optimization/runs/20261010T084344Z-a54afaa6a23c/offline.wav)。實體Stream依使用者指示尚不開；若日後另行恢復，再做Start/Stop、loopback與接收端錄音，不能用Pipeline p95代替裝置延遲。完整重跑指令見 [README](../../tools/realtime_optimization/README.md)。

## 變更檔案與 Git

新增`tools/realtime_optimization/`（benchmark、instrumentation、metrics、quality、f0_evaluation、preset、README）、`tests/test_realtime_benchmark.py`、三份文件與evidence；`.gitignore`排除runs／私人preset，`tests/run_safe_checks.py`只增加新工具編譯。使用者原有AGENTS.md與docs/CUSTOMIZATION.md未改寫。production RVC diff為空，現有設定及量測資產hash保持一致。**沒有新增commit、沒有push**；目前是可供檢視／提交的工作樹。
