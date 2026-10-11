# 第一階段實作紀錄與下一階段計畫

本階段已建立獨立 Benchmark、預設關閉的 Profiling、音訊品質／外部逐字稿介面與既有 RMVPE 的獨立輸出 F0 評估。使用者允許 GPU 後，已完成正常五輪、階段 Events、短段 CUDA trace、同資產 Offline／Realtime WAV 及補充情境。實測、限制與安全回歸見 [baseline_report.md](baseline_report.md)，操作見 [README](../../tools/realtime_optimization/README.md)。不修改 production 推論路徑，不換 RVC 架構。

## 實測優先順序

主要正常結果為 p95 200.86 ms、p99 222.16 ms、2/345 個250ms deadline miss、RTF0.5273。背景GPU已有其他負載，另一次正常單輪p95為167.45ms，不能把差異當最佳化收益。以下排名使用實測分段，不再沿用「HuBERT／Phase Vocoder／Index應最慢」的靜態假說。

| 優先 | 量測證據 | 下一步局部實驗 | 預期收益與限制 | 風險／品質副作用 | 驗證與回滾 |
|---|---|---|---|---|---|
| 1 Generator 的launch／同步路徑 | 診斷host p50 96.90ms；短trace kernel p50僅4.81ms、約1043 kernels/chunk | 先按CPU op細分rate.item、scalar／D2H等待及小kernel啟動，保留模型數學；單獨測試可移出的常數／shape控制 | 純kernel即使減半，只省約2.4ms診斷GPU工作；主要收益需來自launch或等待，不能依host96.9ms承諾全可刪 | scalar控制需保持相同分支／裁切；重排隨機項或同步可能改輸出／SOLA | 固定seed、多次無profile對照、共同有效WAV及shape/latent切片；新旗標預設關閉，獨立commit／停用旗標回滾 |
| 2 RMVPE／F0 的小操作與啟動 | host p50 15.53ms／stream span60.72ms；kernel p50 3.49ms、約832 kernels/chunk | 先追mel／decode／quantization／cache及scalar同步；現有310ms視窗已存在，不再以整段F0重算為理由改寫 | 純kernel減半約1.75ms診斷GPU工作；省launch才可能較大。不得將60.7ms Event跨度全當RMVPE kernel成本 | 不能改F0 contour、V/UV、hop、threshold或coarse buckets；cache frame順序須一致 | 真實RMVPE輸出與cache逐元素、V/UV切換、氣音／高低音WAV；原路徑保留、開關回滾 |
| 3 HuBERT 完整context | host p50 7.74ms；stream17.10ms；另一短trace kernel p50 22.66ms（純GPU第一） | 先看固定shape、allocator／常數重用，context仍保留2.81s；若考慮compile須先確認既有Windows工具鏈與數值行為，不安裝依賴 | 純kernel若減20%，約4.5ms診斷GPU工作；不能直接換算正常端到端收益 | 雙向attention／位置使跨chunk特徵快取與縮context不等價，容易傷中文咬字／聲調 | 相同模型feature誤差、相似音／疑問句／長段、人工盲聽；不改預設context，獨立旗標 |

上表中的假設節省數字是由診斷kernel量級估算，**不是已實作收益或可保證的正常latency改善**；不同pass與背景負載不可相加。若某局部實驗真能在正常pass省下10ms/chunk，250ms設定下RTF才約下降0.04；本次沒有證明該改善存在。採用條件應是重複正常量測改善、deadline尾端沒有惡化、音訊結果／主觀品質均通過。

## 純GPU計算與次要候選

短段trace的純kernel前三為HuBERT、SOLA、Generator，與host模組呼叫排序不同。SOLA correlation kernel p50約13.99ms（兩次合計），是值得獨立研究的局部運算；先在相同crossfade/search範圍比較精確correlation／offset，不要先換phase／crossfade或降低搜尋品質。數值ties或微小差異會改alignment、click與韻律，必須包括onset／靜音切換。

50ms Phase Vocoder的完整診斷host約4.89ms／stream5.96ms，短trace kernel約0.0905ms；O(N²)仍可能在更長crossfade成長，但本case不是首要瓶頸。可評估grid/window重用，避免以此保證顯著收益。其fractional-bin相位合成不能直接用一般iFFT代替。

本case Index為10k vectors、GPU完整檢索、無CPU fallback，診斷stream p50約1.59ms，短trace kernel約0.127ms。暫緩ANN、ONNX檢索或新增快取；改鄰居可能改聲線，且目前不是主要成本。Buffer、Volume Envelope／resampling同樣不是最大的獨立scope；先解釋其residual同步而非全面重寫。

## 音質與穩定性優先

1. **coarse F0 normalization差異已實際CPU重現**：Offline在50/1100Hz為1/255，Realtime為8/246；fineF0相同。影響conditioning，屬品質實驗，不能混入宣稱「不改結果」的效能prototype。下一階段用預設關閉的可切換normalization做同模型A/B，保留原演算法與回滾。
2. **氣音／清音／startup與VAD**：固定preset有13塊startup零輸出；VAD跳過時context不前進，重新起音可能影響F0cache及韻律。已有補充WAV，但沒ASR／人工評分；低音量旗標門檻不能證明沒有截字。先人工標記被吞的字與時間，對照輸入活動及輸出V/UV。
3. **queue backlog／重播**：正常模型曾2次超過服務deadline，實體IPC／callback／scheduler還未量測。先在worker控制路徑以聚合sequence/timestamps確認drop與replay；不得在callback加logging、等待、阻塞queue或控制鎖。保留既有連續性／過期backlog策略。
4. **輸出F0與邊界**：一般聊天的RT中位誤差10.75cents、V/UVagreement93.50%，須考慮60ms近似對齊；無NaN／clipping或>0.1邊界旗標不能代表聽不到pop或自然女聲。GPU相同seed重跑仍有小幅sample差異，先查可重現性，不能以寬容差宣稱數值完全不變。

## 下一階段順序與採用條件

1. 使用者先用現成Offline／Realtime WAV做等響度、共同內容A/B，記中文發音／疑問句聲調、台灣口音、氣音／句尾、金屬感與長段飄移；不必再手動完成本次GPUbenchmark。
2. 在固定背景情境下增加正常pass樣本，分開空閒／遊戲負載；不由工具停止或改使用者程序。先縮小Generator/F0的CPU op／同步來源，另量SOLA精確correlation。與kernel、Event、host三種數字分開解讀。
3. 選一個有量測支持、可證明不改模型運算的局部prototype，預設關閉，同seed及參數重跑；若真實GPU非bitwise，先建立原路徑多次變動基準與解釋，再決定合理驗證，保留WAV供盲聽。沒有可靠收益就不採用。
4. F0coarse差異另開品質A/B，不與效能prototype同時修改，避免無法歸因。ASR若沒有現成依賴／權重，仍用外部逐字稿介面，不自行下載大型模型。
5. **實體Stream目前禁止**。日後另行恢復後才做Start/Stop、loopback、Discord接收端／OBS錄音及遊戲負載，記裝置Device Name + Host API、sample rate、driver buffer及共同時鐘；Pipeline p95保持獨立欄位。

torch.compile、FP16、CUDA Graph、ONNX或改Index backend均涉及shape、動態控制、隨機latent與音質風險，沒有當作第一階段通用開關。本階段沒有production prototype或前後加速比較，只有既有路徑的baseline與診斷。

## 保護與交付

工具／測試／文件為同工作樹獨立新增，production RVC原始碼無差異，既有weights／Index／測試WAV／設定hash保留。未建立缺模型的worktree，未切branch、stash/reset、執行installer、安裝依賴或開Stream。使用者尚未要求commit，因此沒有新commit或push。

回滾只涉及新工具／測試／文件與兩處小型整合（gitignore、safe_checks編譯清單）；不自動刪runs或任何錄音／權重。原有AGENTS.md、CUSTOMIZATION.md與先前客製化保持不變。詳細檔案、原始JSON/CSV/WAV與實際測試狀態見baseline。
