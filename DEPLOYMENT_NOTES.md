# Qwen-Image-2.1 於 NVIDIA DGX Spark 部署實戰全紀錄與技術筆記

本文件詳細紀錄 **Qwen-Image-2.1** 多模態擴散模型於 **NVIDIA DGX Spark** 桌面超級電腦上的完整落地實施過程。內容涵蓋初始需求分析、硬體規格與資源探勘、核心技術踩坑與除錯、前端「以圖改圖」機制破解、各大論壇調優規範，以及最終的服務運行結果。

---

## 目錄
1. [起點：需求分析與初始方案](#1-起點需求分析與初始方案)
2. [探勘：DGX Spark 實機規格與生態盤點](#2-探勘dgx-spark-實機規格與生態盤點)
3. [過程：部署實施與關鍵踩坑修復全紀錄](#3-過程部署實施與關鍵踩坑修復全紀錄)
   - 3.1 [記憶體釋放與多服務共存策略](#31-記憶體釋放與多服務共存策略)
   - 3.2 [ARM64 (aarch64) 容器與依賴地獄排除](#32-arm64-aarch64-容器與依賴地獄排除)
   - 3.3 [模型檔案探勘與高速下載管線](#33-模型檔案探勘與高速下載管線)
   - 3.4 [「以圖改圖」機制盲點排查與破解](#34-以圖改圖機制盲點排查與破解)
   - 3.5 [各大開源論壇調優與參數審查 (CFG / Sampler)](#35-各大開源論壇調優與參數審查-cfg--sampler)
4. [成果：最終部署架構與操作指南](#4-成果最終部署架構與操作指南)
5. [總結與維運守則](#5-總結與維運守則)

---

## 1. 起點：需求分析與初始方案

### 1.1 任務目標
* **目標模型**：Qwen-Image-2.1（由阿里巴巴開源之 7B Single-Stream DiT + Qwen3-VL-8B 複合圖像生成與編輯模型，支援 2K 原生解析度與 RGBA 原生透明通道）。
* **目標硬體**：NVIDIA DGX Spark。
* **交付要求**：規劃完整的端到端部署方案，包含支援文字生圖、以圖改圖、指令局部編輯的可操作前端 WebUI，並實現系統永久常駐運維。

---

## 2. 探勘：DGX Spark 實機規格與生態盤點

在初始規劃階段，曾一度將其誤認為傳統多卡資料中心架構，經即時網路查核與遠端連線實測，確立了這台機器的精確特性：

### 2.1 硬體規格確認
* **主晶片**：NVIDIA **GB10 Grace Blackwell 超級晶片**（整合 20 核心 ARM64 Grace CPU + Blackwell GPU 架構）。
* **記憶體架構**：**128 GB 統一共享記憶體 (Unified Memory)**。
* **頻寬特性**：256-bit LPDDR5x，頻寬約 **273 GB/s**（非資料中心 HBM3e 的 3,000+ GB/s）。
* **作業系統**：NVIDIA DGX OS (Ubuntu 24.04 aarch64，Linux 核心 6.17.0-1029-nvidia)。

### 2.2 主機運行生態與資源瓶頸審查
透過 Tailscale 網路（目標 IP: `<DGX_HOST>` / `<DGX_HOSTNAME>`）進行遠端實況盤查：
1. **多模型佔用實況**：
   * `qwen3.6-nvfp4`（Port 8002）：鎖定 15GB KV-Cache，實際佔用約 **39.5 GiB** 記憶體。
   * `qwen3.8-nvfp4`（Port 8006）：鎖定 10.7GB KV-Cache，實際佔用約 **37.5 GiB** 記憶體。
   * 其他服務：LiteLLM Gateway (Port 4000)、Hermes Agent (Port 9118)、本地桌面 Chromium 等。
2. **記憶體餘裕警訊**：
   * 121 GiB 總量中已被佔用 **90 GiB**，Swap 已吃掉 10 GiB，**實質可用僅剩 31 GiB**。
   * **模型真實開銷**：Qwen-Image-2.1 並非只有 7B，其真實架構為 **7B DiT + 8B Qwen3-VL + 64 通道 2K VAE**，若以原生 BF16 載入，峰值需求達 40~45 GB，在原本環境下將直接觸發致命 OOM。

---

## 3. 過程：部署實施與關鍵踩坑修復全紀錄

### 3.1 記憶體釋放與多服務共存策略
* **操作**：依指令暫時停止 `qwen3.6-nvfp4`（Port 8002）容器：
  ```bash
  docker stop qwen3.6-nvfp4
  ```
* **成效**：瞬間釋放出 **46 GiB** 記憶體，使系統可用記憶體躍升至 **76 GiB**，為 Qwen-Image-2.1 提供了充沛安全的運行空間。

### 3.2 ARM64 (aarch64) 容器與依賴地獄排除
* **基礎鏡像選型**：採用官方已針對 Blackwell 及 ARM64 編譯完備的 `nvcr.io/nvidia/pytorch:25.11-py3`。
* **踩坑 1：PyPI 無預編譯 Wheel**
  * 在 ARM64 上執行 `pip install flash-attn` 會失敗或觸發數小時本機編譯。
  * **解法**：全面改用 PyTorch 2.10 原生 **SDPA (Scaled Dot Product Attention)**，由 CUDA 13.2 直接調用 Blackwell Tensor Core。
* **踩坑 2：Torchaudio ABI 衝突**
  * ComfyUI 在啟動時導入了預編譯的 `torchaudio`，觸發 `undefined symbol: torch_library_impl` 崩潰。
  * **解法**：Qwen-Image-2.1 為純影像模型，透過 Docker 移除 `torchaudio`，並對其依賴的音訊相關 stub 檔案進行隔離，順利讓 ComfyUI 核心啟動。

### 3.3 模型檔案探勘與高速下載管線
* **權重選型**：採用官方推薦之 `Comfy-Org/Qwen-Image-2.1` 標準拆分格式：
  1. `vae/qwen_image_2.1_vae_bf16.safetensors` (0.63 GB / 645 MB)
  2. `diffusion_models/qwen_image_2.1_bf16.safetensors` (13.25 GB / 14 GB)
  3. `text_encoders/qwen3vl_8b_bf16.safetensors` (16.33 GB / 17 GB)
* **傳輸加速**：初期以原生單線程下載僅有 0.09 MB/s；透過在 Docker 容器內注入 Rust 高速下載器 `hf-transfer` 與 Hugging Face Token，將頻寬提速至 **6 ~ 8 MB/s**，於 40 分鐘內完成 30.2 GB 的完整權重下載。

### 3.4 「以圖改圖」機制盲點排查與破解
在初版 Gradio 前端上線後，使用者測試回報：**「上傳圖片後，模型不懂修改原圖，反而畫了一張全新的圖片」**。

經深入分析與源碼審查，揭露了兩個根本原因：
1. **初始畫布 (Latent) 錯誤**：
   * 舊版工作流在取樣器使用了 `EmptyLatentImage`（全黑空畫布），且 `denoise = 1.0`。
   * 在數學上，取樣器從純高斯噪聲出發，即使模型看過參考圖，也只能重新構圖畫一張新圖，無法保留原圖幾何輪廓。
2. **缺少官方錨點標記 `<image1>`**：
   * Qwen-Image-2.1 的文字編碼器（`TextEncodeQwenImage21`）要求提示詞必須包含 `<image1>` 或 `<image_1>`，否則語意注意力不會作用於該圖片。
3. **終極修復方案（VAEEncode + 原圖 Latent）**：
   * 上傳的原圖直接經由 **`VAEEncode`** 轉換為初始 Latent 輸入給 KSampler。
   * 加入 **`Denoise` 重繪強度滑桿 (0.3 ~ 0.8)**，實現精確的局部重繪與風格轉換。

### 3.5 各大開源論壇調優與參數審查 (CFG / Sampler)
檢索 Reddit、Civitai 與 ComfyUI 官方 Issue 中全球開發者的實測反饋，對我們的預設值進行了審查與矯正：
* **CFG Scale（引導係數）**：
  * **原設定**：5.0
  * **社群黃金標準**：**`3.5 ~ 4.0`**（Qwen3-VL 語意敏銳度極高，超過 4.5 會導致皮膚出現塑料假面感與光影過曝）。
  * **改善**：全面將預設值調整為 `3.5`。
* **負向提示詞 (Negative Prompt)**：
  * DiT 架構對長負向詞極為敏感，過多標籤會稀釋正向詞權重。
  * **改善**：精簡為 `blurry, distorted`，並註明改圖時可直接留空。
* **取樣器搭配**：維持社群一致公認最穩定之 **Euler + Simple Scheduler**。

---

## 4. 成果：最終部署架構與操作指南

目前於 DGX Spark 主機上已全面上線兩套互補的服務前端：

```
                              [ 使用者終端瀏覽器 ]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼ (HTTP / WebSocket)                                ▼ (HTTP / WebSocket)
   【Port 7860: 專用 Gradio WebUI】                     【Port 8188: ComfyUI 工作台】
   • 支援以圖改圖 (VAEEncode 重繪)                       • 完整節點式自訂編排
   • Denoise 重繪幅度控制 (0.3~0.8)                     • 支援複雜 ControlNet / Lora
   • CFG 3.5 黃金出圖預設                              • 原生 TextEncodeQwenImage21 節點
   • 棋盤格透明背景預覽 (RGBA)                          • 支援高解析度圖層拆解
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
             【NVIDIA DGX Spark 核心 (<DGX_HOST>)】
             • 晶片: GB10 Grace Blackwell (20-Core ARM64)
             • 精度: fp8_e4m3fn_fast (適配 LPDDR5x 頻寬)
             • 模型: 7B DiT + 8B VL + 64-Channel RGBA VAE
             • 運維: Docker Compose (restart: always)
```

### 4.1 服務訪問網址
* **專用輕量 WebUI（推薦一般使用）**：👉 **[http://<DGX_HOST>:7860](http://<DGX_HOST>:7860)**
* **ComfyUI 專業工作台（推薦進階設計）**：👉 **[http://<DGX_HOST>:8188](http://<DGX_HOST>:8188)**

### 4.2 以圖改圖操作指引
1. 開啟 `http://<DGX_HOST>:7860`，模式選擇 **「以圖改圖 / 畫圈局部替換」**。
2. 上傳欲修改的原圖至互動畫布。
3. 依照目標調節 **Denoise (重繪強度)**：
   * **0.35 ~ 0.45**：微調、改色、微修細節（極致保持原樣）。
   * **0.55 ~ 0.65（推薦）**：換裝、換背景、換髮型（主體姿勢與輪廓完全鎖定）。
   * **0.75 ~ 0.85**：大幅改變畫風（轉為油畫、賽博龐克、動漫風）。
4. 輸入指令（如：`將衣服換成黑色西裝，背景換成純白`），點擊 **開始執行生成 / 替換**。

### 4.3 畫圈替換與局部重繪技術事實 (Official Qwen-Image-2.1 Architecture)
依據阿里官方開源 GitHub（`QwenLM/Qwen-Image-2.1`）與 ComfyUI 核心實作代碼（`comfy/ldm/qwen_image21/model.py`）：
1. **非傳統剪貼拼貼（Not Cut-and-Paste）**：
   - 傳統 Photoshop 剪貼是指將圖像局部裁切（Crop）獨立生成後再邊緣貼回，容易產生透視斷裂與光影違和。
   - Qwen-Image-2.1 採用 **Single-Stream MMDiT（單流多模態擴散 Transformer）** 架構，在 `build_sequence` 函數中將文字指令、參考圖像 Latent 與生成畫布拼成一個單一序列：
     ```python
     # comfy/ldm/qwen_image21/model.py (build_sequence)
     # 序列結構: [Text Tokens] + [Ref Latent Tokens] + [Target Image x Latent Tokens]
     hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, image_slots)
     ```
   - 在 32 層 DiT 中，目標像素在每一去噪步都透過全雙向 Self-Attention 即時感知全圖環境光、陰影反光與身體結構；配合 **Prefix KV Cache Reuse**，使新生成的物體從潛空間源頭自然生長融入，而非事後貼圖。
2. **三種官方局部編輯形式**：
   - **Circles（畫圈標記）**：直接在圖像上畫彩色圈，由內建 **Qwen3-VL (8B)** 視覺大模型透過空間視覺引導（Visual Grounding）理解圈內主體。
   - **Painted Annotations（筆刷塗抹）**：在目標區域粗略填色，提供形狀與色彩先驗。
   - **Separate Masks（二值遮罩）**：透過 ComfyUI 的 `VAEEncodeForInpaint` 進行 Latent 遮罩鎖定，圈外像素 100% 絕對凍結，圈內去噪並自動進行邊緣羽化（Grow Mask）。

### 4.4 升級版前端互動操作
- **互動畫布（`gr.ImageEditor`）**：支援紅（`#ff0000`）、綠（`#00ff00`）、藍（`#0000ff`）、白（`#ffffff`）彩色筆刷與橡皮擦，可自由調節筆刷粗細。
- **編輯模式切換**：
  1. `🔒 遮罩局部精準重繪 (Mask Inpaint)`：以筆刷塗抹物體，圈外 100% 凍結，圈內依指令替換（推薦換髮型、換頭盔、換飾品）。
  2. `🎯 畫圈視覺引導 (Circle Grounding)`：隨手畫圈，帶圈影像直接傳遞給 Qwen3-VL 識別重塑。
  3. `📎 第二參考圖 (Multi-Reference)`：支援上傳衣服或商品，執行跨圖穿戴（`<image1>` + `<image2>`）。

### 4.5 多圈選與多遮罩定位規範 (Multi-Circle / Multi-Mask Rules)
當畫面中有多個目標需要同時修改時，模型支援以下兩種定位語法：

1. **畫圈模式（以「顏色」精準綁定目標）**：
   - 使用不同顏色筆刷圈出不同目標，在 Prompt 中明確指定顏色：
   - **官方標準範例**：
     > *"Remove the metal watch in the blue circle, change the hair in the red circle to black, and replace the area in the green circle with gray short-sleeved linen pajamas."*  
     > *(移除藍圈裡的金屬手錶，將紅圈裡的頭髮換成黑色，並將綠圈裡的區域替換為灰色短袖亞麻睡衣。)*
   - **語法結構**：`將 [顏色] 圈內的 [原物件] 替換成 [新物件]` 或 `移除 [顏色] 圈內的 [物件]`。

2. **遮罩重繪模式（以「解剖/空間語義」定位）**：
   - 遮罩二值化後不帶顏色，依靠 Qwen-Image 對人體解剖與場景語義的空間理解：
   - **提示詞寫法**：直接並列描述各部位目標，例如：  
     > *「將人物的頭部換成太空人頭盔，身上的衣服換成黑色西裝，背景嚴格保持不變。」*  
     *(模型會自動將頭盔填入頭部遮罩，西裝填入身體遮罩)*

### 4.6 前端升級變更紀錄 (Changelog)
- **2026-09-22 升級**：
  - 由原本只支援整圖上傳的 `gr.Image` 升級為 Gradio 6 現代畫布 `gr.ImageEditor`。
  - 新增 `parse_editor_data` 函數，自動萃取 `background`、`composite` 及 `layers` 的 Alpha 遮罩。
  - 後端 ComfyUI 工作流擴展支援 `LoadImageMask` + `VAEEncodeForInpaint`（grow_mask_by=6）。
  - 同步檔案至 `/home/<SSH_USER>/qwen-image/gradio_app.py` 並重啟 `qwen-gradio-ui` 容器驗證通過。

### 4.7 剪貼簿貼上 (Clipboard Paste) 完整支援實作
針對所有圖片輸入組件（主畫布 `editor_input` 與第二參考圖 `ref_image_2`）：
1. **組件級剪貼簿原生整合**：
   - 明確配置 `sources=["upload", "clipboard"]`，組件右上角提供獨立的「剪貼簿貼上」圖示，並移除無用的視訊攝影機按鈕。
2. **三種貼上操作方式**：
   - **方式一（組件點選貼上）**：滑鼠點擊圖片上傳區域，按下鍵盤 `Ctrl + V`（或 Mac `Cmd + V`）。
   - **方式二（剪貼簿按鈕）**：點擊組件上的剪貼簿圖示，直接自系統剪貼簿載入圖片。
   - **方式三（全域快捷貼上）**：注入 `paste_helper_script`，當用戶在頁面任意位置按下 `Ctrl + V`（且焦點不在文字提示詞輸入框時），自動捕捉剪貼簿中的截圖（如 Windows `Win + Shift + S` 或網頁複製的圖片）並注入主畫布。

### 4.8 官方 DEMO 多參考圖庫 (1~10 張) 與智能指向貼上
為完全對齊阿里官方 DEMO（HuggingFace Space `Qwen/Qwen-Image-2.1`）之多圖融合能力，並徹底解決參考圖無法貼上的問題：
1. **多參考圖集管理庫（Gallery + State）**：
   - 提供專屬「📋 點此處按 Ctrl+V 貼上圖片」紫色虛線接收框。
   - 用戶只需複製圖片並在該框按下 `Ctrl+V`，圖片會**自動依序追加至右側圖庫**，並即時自動清空接收框，讓用戶可**連續複製貼上多達 10 張參考圖**。
   - 圖庫中清晰標註序號標籤：`<image1>`、`<image2>`、`<image3>`...，提示詞可直接精準引用。
   - 支援拖曳/多選檔案批次匯入，並附帶「🗑️ 清空所有參考圖」與「⏪ 移除最後一張」快捷管理功能。
2. **滑鼠指向智能路由（Hover-Aware Paste Routing）**：
   - 解決了原本全域貼上腳本強制攔截並鎖定在單一輸入框的 Bug。
   - 瀏覽器端動態追蹤滑鼠位置（`window.lastHoverBox`）：
     - 當滑鼠移至「參考圖區」按 `Ctrl+V` ➔ **100% 精準貼入參考圖庫**。
     - 當滑鼠移至「主畫布區」按 `Ctrl+V` ➔ **100% 精準貼入主畫布**。
3. **後端 ComfyUI 多圖拓撲**：
   - 動態遍歷上傳之參考圖，依序註冊 `LoadImage` 節點注入 `TextEncodeQwenImage21` 的 `images: {"image_1": ..., "image_2": ..., ...}`。
   - 多圖合成模式下，直接調用模型專屬之第 3 輸出埠 `["4", 2]` 作為 Latent 畫布，完美實現多角色、多商品跨圖無縫融合。

### 4.9 官方倉庫 100% 對齊變更 (Official Alignment Changelog)
依據阿里官方開源倉庫 `https://github.com/QwenLM/Qwen-Image-2.1` 與 `prompt_rewrite/prompts/system_prompt_edit.txt`：
1. **CFG 與負向提示詞參數校正**：
   - 將 CFG 預設值從 3.5 校準為 **`1.0`**（官方 Diffusers `cfg_scale=1.0` 與 ComfyUI 官方模板黃金值）。
   - 負向提示詞預設為空字串 `""`，消除 DiT 模型在 CFG > 1.0 時因雙倍負向去噪引起的特徵漂移與風格失真。
2. **空心圈自動孔洞填充 (Anti-Jagged Hole-Filling)**：
   - 整合 `scipy.ndimage.binary_closing` 與 `binary_fill_holes`。
   - 當使用者在主圖上畫空心圈時，後端自動將圈內實心填滿為完整二值遮罩，從根源徹底消除「只有細圓線被重繪、中間露出原本人臉」的鋸齒斷裂與鬼影缺陷。
3. **畫圈視覺引導乾淨畫布重塑**：
   - 帶圈圖（`composite`）僅送交 Qwen3-VL 識別空間座標；Latent 編碼器改由乾淨原圖（`clean_bg`）產生，徹底防止彩色筆跡墨水在重繪時產生色斑外溢。
4. **官方提示詞範本一鍵注入**：
   - 內建符合 `system_prompt_edit.txt` 規範的標準 Prompt 範本（包含五圖穿搭寫真、雙色圈選替換、單件衣服換穿、主體場景融合）。

---

## 5. 總結與維運守則

1. **資源互斥維護**：
   * 目前暫停的 `qwen3.6-nvfp4`（Port 8002）可視業務需要隨時透過 `docker start qwen3.6-nvfp4` 喚醒；但在 DGX Spark 記憶體滿載時，建議不同時對 Qwen-Image 進行大批次並發推論。
2. **服務重啟與維護指令**：
   ```bash
   # 查看當前運行容器
   docker ps --filter name=qwen
   
   # 重啟 Gradio 介面
   docker restart qwen-gradio-ui
   
   # 查看 ComfyUI 後端即時日誌
   docker logs -f qwen-image-comfyui
   ```
3. **檔案位置**：
   * 模型路徑：`/home/<SSH_USER>/qwen-image/ComfyUI/models/`
   * 前端腳本：`/home/<SSH_USER>/qwen-image/gradio_app.py`
   * 服務定義：`/home/<SSH_USER>/qwen-image/docker-compose.yml`

---

## 6. 【2026-09-22 重大發現】參考圖從未生效：ComfyUI autogrow 動態輸入的 API 缺陷

### 6.1 症狀
「以圖改圖」「多圖參考融合」「畫圈視覺引導」三種模式**全部形同虛設**：
上傳的圖片從未進入模型，實際執行的只是帶著那段文字的純文字生圖。
表現為輸出一張構圖無關的新圖，但 prompt 的文字語意（服裝、風格）有生效。

### 6.2 根因
`TextEncodeQwenImage21` 的 `images` 輸入型別是 `COMFY_AUTOGROW_V3`（動態增長輸入）。
ComfyUI 0.37.0 在 `/prompt` API 路徑上無法正確接收此型別：

| 傳法 | 結果 |
| :--- | :--- |
| 平鋪 `"image_1": ["10",0]` | `TypeError: execute() got an unexpected keyword argument 'image_1'` |
| 巢狀 `"images": {"image_1": ["10",0]}` | 通過驗證但**被靜默丟棄** |

原因見 `comfy_api/latest/_io.py:1194` `_expand_schema_for_dynamic()`：
schema 展開時「purposely do not include self in out_dict」，對外合法的輸入名是
`image_1`…`image_16`，**沒有 `images`**；而把平鋪名收攏回 `images` 的
`execution.py:294` `build_nested_inputs()` 依賴 `v3_data["dynamic_paths"]`，
該對應表只有網頁前端送 workflow 時才會產生，直接打 API 不會有。

### 6.3 診斷關鍵：輸出尺寸
節點原始碼 `comfy_extras/nodes_qwen.py:148`：
```python
latent_w = latent_h = resolution or 1024   # 無參考圖時的預設正方形
for name in sorted(images, ...):
    ...
    if not images_vl:
        latent_w, latent_h = width, height  # latent 跟隨第一張參考圖
```
**輸出恆為 1024x1024 正方形 = 參考圖迴圈一次都沒跑。**
輸入 896x1152 而輸出 896x1152，才代表參考圖真的生效。這是最快的自我檢查方式。

### 6.4 踩坑：不可用 md5 比對輸出
ComfyUI 會把整份 workflow JSON 寫進 PNG metadata，送出的 JSON 不同則檔案位元必然不同。
曾因此誤判「參考圖有作用」。**必須比對像素**（`np.abs(a-b).max()`），
實測「有送 images」與「完全不送」兩張圖**逐像素完全相同**，才確立參考圖被丟棄。

### 6.5 解法：TextEncodeQwenImage21API 轉接節點
新增 `ComfyUI/custom_nodes/qwen21_api_shim.py`，提供普通的 `image1`…`image10` 輸入，
內部**直接呼叫官方 `TextEncodeQwenImage21.execute()`**，不改任何演算法，純傳輸層轉接。

`gradio_app.py` 對應改動：
- `class_type` 改為 `TextEncodeQwenImage21API`
- `images` 巢狀 dict 改為平鋪 `image1` / `image2` / … 直接放在 `inputs`
- `resolution` 由 `0` 改回節點預設 `1024`

### 6.6 驗證結果
官方 ComfyUI 模板換衣範例（`portrait_model_denim.png` + `clothing_light_blue_denim_shirt.png`，
官方 seed `1070478148268574`，40 步、cfg 1.0、bf16）：

| | 修正前 | 修正後 |
| :--- | :--- | :--- |
| 輸出尺寸 | 1024x1024（錯） | **896x1152（正確，跟隨 image_1）** |
| 人物身分 | 完全不同的人 | **同一人、同姿勢、同側臉** |
| 背景光線 | 重新生成 | **保留原背景與打光** |
| 衣服 | 有換（僅文字生效） | 正確換成 image2 的牛仔襯衫 |

### 6.7 教訓
先前一直歸因於「官方 edit 路徑本來就是整張重生成」是**錯誤解釋**。
Qwen-Image-2.1 的編輯能力完全正常，是傳輸層把圖擋掉了。
排查順序應為：**先確認輸入真的有進到模型（看輸出尺寸），再談參數與品質調校。**

---

## 7. 【2026-09-22】DGX Spark 統一記憶體：ComfyUI 權重雙倍載入修正

### 7.1 問題
DGX Spark 上 safetensors 經 mmap 載入會在 RAM 留一份常駐副本，`.to(device)` 再複製一份到
「VRAM」—— 但 Spark 的 VRAM 就是 RAM，於是同一份權重存在兩份，128GB 實際只能當 64GB 用。
來源：<https://forums.developer.nvidia.com/t/buyers-beware-dgx-spark-limited-to-64gb-in-comfyui/356573>

### 7.2 修正（已寫入 docker-compose.yml）
```yaml
environment:
  - PYTORCH_NO_CUDA_MEMORY_CACHING=1
command: python3 main.py --listen 0.0.0.0 --port 8188 --highvram --disable-mmap --disable-pinned-memory
```
`PYTORCH_NO_CUDA_MEMORY_CACHING` 與 `--disable-pinned-memory` 取自
<https://github.com/Triplany/comfyui-dgx-spark>（GB10 sm_121 aarch64 專用優化）。

### 7.3 實測效果（官方換衣範例，冷載入 + 25 步）
| 指標 | 改前 | 改後 |
| :--- | :--- | :--- |
| 冷載入 + 生成 | 231 秒 | **83 秒（快 2.8 倍）** |
| ComfyUI 常駐 | 31.5 GB | 29.1 GB |
| swap 峰值 | 15/15 全滿 | 11/15，未飆滿 |
| buff/cache | 被壓到 1 GB | 30 GB |

速度提升的主因不只是省記憶體，更是省掉那次多餘的 31.5GB 權重複製 ——
GB10 頻寬僅 273GB/s，這個複製非常昂貴。

### 7.4 附帶結論
同一專案實測 SageAttention 2.2 相對原生 PyTorch SDPA 僅差 1~2%，其預設關閉。
本部署沿用 §3.2 的 SDPA 決定即可，不需為 SageAttention 面對 ARM64 編譯問題。
