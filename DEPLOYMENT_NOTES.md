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

---

## 8. 【2026-09-23】權重精度 A/B 實測：int8_convrot vs bf16

### 8.1 緣起
第 6、7 節修正後，系統已可正常運作，但仍有兩個待決問題：
1. 先前下載的 `int8_convrot` 權重（官方 ComfyUI 模板的預設值）一直未啟用，不確定值不值得切換。
2. 官方建議的文生圖 prompt 改寫模型 `PE-T2I` 需要約 19GB 常駐記憶體，
   而當時 bf16 影像模型佔約 30GB，加上 `qwen3.8-nvfp4`（37.5GB）與 PE-I2I（20GB）已無餘裕。

兩者互相牽連：**若 int8 可用，省下的記憶體正好讓 PE-T2I 進駐。** 故以實測決定。

### 8.2 一個必須先破除的誤解
`int8_convrot` 與先前造成畫質降級的 `fp8_e4m3fn_fast` **不是同一回事**：
- `fp8_e4m3fn_fast`：載入時把 bf16 權重即時硬轉，單純數值截斷，無任何補償。
- `int8_convrot`：官方**離線**量化的獨立權重檔，`convrot` 指旋轉式量化
  （rotation-based，同 QuaRot / SpinQuant 家族），先以正交旋轉攤平離群值再量化。

### 8.3 GB10 低精度矩陣運算實測（4096³ matmul，PyTorch 2.10，sm_121）
| 精度 | 耗時 | 吞吐 | 相對 bf16 |
| :--- | ---: | ---: | ---: |
| bf16 | 4.63 ms | 29.7 TFLOPS | 1.00x |
| int8 | 9.12 ms | 15.1 TOPS | **0.51x** |
| fp8 | 7.23 ms | 19.0 TFLOPS | **0.64x** |

**GB10 的低精度矩陣運算沒有加速，反而更慢。** 但這不影響結論，因為 ComfyUI 採
**僅權重量化**：運算仍走 bf16，量化只減少權重搬運量。GB10 頻寬僅 273GB/s，
搬運才是瓶頸 —— 所以 int8 仍然更快。

此結果也回頭說明：§3.5 宣稱 fp8 讓「每步僅 ~1.4 秒」缺乏對照組，
當初用 fp8 犧牲畫質，很可能連速度都沒換到。

### 8.4 端到端 A/B（官方換衣範例，官方 seed，40 步）
| 指標 | bf16 | int8_convrot | 差異 |
| :--- | ---: | ---: | :--- |
| 熱機推論 | 63.9 s | **54.8 s** | int8 快 14% |
| 冷載入 | 163.0 s | **72.4 s** | int8 快 2.25 倍 |
| 常駐記憶體 | ~30 GB | **16.9 GB** | 省 14 GB |
| 文字渲染（霓虹招牌 2K） | 正確 | **正確** | 無退步 |
| 身分保真（換衣） | 良好 | **良好** | 無退步 |

### 8.5 量測踩坑
首次以 2K 文生圖比較時得到「int8 略慢」（240.5s vs 236.1s）的錯誤結論。
原因是該次為 int8 檔案**首次自 NVMe 讀取**，無 page cache，被載入時間污染。
**乾淨的比較必須用熱機數據**（權重已在記憶體中）。

### 8.6 結論與處置
int8_convrot 三項指標全勝且畫質無退步，已將 WebUI 預設改為 int8，
並保留 bf16 為可選項（`WEIGHT_PRESETS`，UI 右側「⚙️ 權重精度」下拉選單）。
省下的 14GB 用於 `PE-T2I` 進駐。

---

## 9. 【2026-09-23】官方 Prompt Enhancer 完整上線（PE-I2I + PE-T2I）

### 9.1 為什麼需要
官方 README 明示：「For best results, we recommend using the official prompt
rewriting models」。**官網 showcase 的 prompt 幾乎都經過擴寫**，
拿短句去比對 demo 本來就贏不了 —— 這不是模型能力差距，是輸入品質差距。

官方提供兩個 checkpoint，**各有自己的 system prompt 與訓練目標，不可互換**
（指向原版 Qwen3.5-VL 9B 會載入也會生成，但多數回應 `parse_ok: false`）：
| task | checkpoint | 輸入 | 回傳欄位 |
| :--- | :--- | :--- | :--- |
| `t2i` | `Qwen/Qwen-Image-2.1-PE-T2I` | 純文字 | `rewritten_prompt`, `wh_ratio` |
| `edit` | `Qwen/Qwen-Image-2.1-PE-I2I` | 文字 + 1~N 張圖 | `rewritten_prompt`, `wh_ratio`, `ratio_follow` |

### 9.2 部署方式
容器內無 vLLM（ARM64），故走官方 `run_transformers.py` 參考實作。
新增 `app/pe_server.py`：常駐 HTTP 服務（127.0.0.1:8200），模型只載入一次。
**推論邏輯完全複用官方程式碼** —— 直接 import `pe_core` 與 `run_transformers`，
沿用其 `rewrite()`、`PresencePenalty`、`build_messages`、`parse_answer` 與 profile 取樣參數
（t2i: presence_penalty 1.5 / max_new_tokens 16256；edit: 0.0 / 24000；皆 `enable_thinking=True`）。
本檔只負責「載入一次 + HTTP 介面」，不改任何演算法。

**踩坑**：載入必須照官方寫法 `from_pretrained(..., low_cpu_mem_usage=True).to(device)`。
官方原始碼註解說明這是為了避免產生全精度 CPU 副本 —— 在統一記憶體機器上，
若改用 `device_map=`，載入瞬間會多吃一份 18GB。

### 9.3 驗證結果
| | PE-I2I (edit) | PE-T2I (t2i) |
| :--- | :--- | :--- |
| `parse_ok` | **True** | **True** |
| 單次改寫耗時 | 144.3 s | 159.7 s |
| 模型載入 | 134.7 s | 約 135 s |
| 回傳欄位 | `ratio_follow: <image1>` | `wh_ratio: 1:1` |

**語言行為差異**（符合兩份 system prompt 的設計）：
- edit 版跟隨使用者輸入語言（中文指令 → 中文描述）
- t2i 版一律輸出英文長段落

實例：t2i 輸入「一隻貓坐在窗邊」6 個字，輸出四段約 2,700 字的英文描述，
涵蓋虎斑紋路、逆光鑲邊、窗框滑軌接縫、玻璃灰塵光點、窗外散景層次、
淺景深、黃金時刻光向，以及完整色盤指定。

### 9.4 前端整合
WebUI 按鈕「✨ 官方 Prompt 改寫 (PE)」：
- 文生圖 → `task=t2i`，回傳的 `wh_ratio` **自動切換尺寸選單**
- 改圖 → `task=edit`，主圖 + 全部參考圖一併送出；**若使用者有手繪，送的是帶標註的圖**
  （對應官方 local editing 支援的 circles / painted annotations）
- `parse_ok: false` 會明確標示，不假裝成功

### 9.5 記憶體事故與修正：PE 服務改為單模型常駐

**事故**：`pe_server.py` 初版設計為「每個 task 各自快取模型、永不卸載」。
當兩個 task 都被使用時，PE 行程實測衝上 **30GB 以上**，帳面總計：

| 元件 | 記憶體 |
| :--- | ---: |
| vLLM `qwen3.8-nvfp4` | 37.5 GB |
| ComfyUI + int8 影像模型 | 16.9 GB |
| PE 兩個模型 | ~38 GB |
| **合計** | **~92 GB / 121 GB** |

帳面尚有餘裕，但統一記憶體架構下 GPU 配置、CUDA context 與 page cache 共用同一池，
載入期還需額外暫存 —— 結果 **swap 被榨到 15/15 全滿、buff/cache 壓到 0**，
系統大量換頁而明顯停滯。終止 PE 服務後立即恢復（已用 99GB → 63GB）。

**修正**：改為**同時只保留一個模型**，載入新 task 前先卸載另一個
（`_unload()`：`gc.collect()` + `torch.cuda.empty_cache()` + `synchronize()`）。

**驗證**：連續呼叫 t2i → edit，記憶體完全未疊加。

| 階段 | used | available | PE 行程 |
| :--- | ---: | ---: | ---: |
| 起始 | 63 GB | 57 GB | — |
| t2i 後 | 82 GB | 39 GB | 18.3 GB |
| edit 後（已切換） | 82 GB | 39 GB | **18.3 GB** |

日誌：`loaded t2i in 106.9s` → `unloaded t2i` → `loaded edit in 94.7s`，
兩次 `parse_ok: True`。代價為切換 task 時多約 95 秒重新載入，同一 task 連續使用不受影響。

**教訓**：在統一記憶體機器上，「多個大模型各自常駐」的直覺式快取設計很危險。
帳面餘裕不等於實際餘裕，應以單一模型為上限並顯式卸載。

---

## 10. 【2026-09-23】前端精簡：模式改為分頁，以及一個靜默的折疊陷阱

### 10.1 模式選擇器改為分頁
原本以 `gr.Radio` 選擇三種模式，判斷方式是比對長字串
（如 `mode == "🎯 以圖改圖 (Image Edit)"`，全檔共 12 處）。
改標題文字時必須同步修改所有比對處，漏一處即靜默失效 —— 先前改名時已踩過。

改為 `gr.Tabs` + `gr.State`，並以穩定識別碼 `MODE_EDIT` / `MODE_MULTI` / `MODE_T2I` 取代字串比對。
**畫布與參考圖庫為三模式共用，故留在分頁之外**，沿用既有顯示/隱藏邏輯，
避免把共用元件複製進各分頁而衍生狀態同步問題。

### 10.2 折疊陷阱（重要）
把所有設定區塊改為 `open=False` 後，出現靜默失效：

| 提交內容 | 折疊時 | 展開後 |
| :--- | :--- | :--- |
| `images` | `[]` | `['image1', 'image2']` |
| `seed` | 127764061（隨機） | 1070478148268574（官方值） |

**原因：Gradio 6 的折疊 Accordion 不渲染子元件，事件提交時送出預設值而非畫面上的值。**
UI 顯示 seed 為官方值，實際送出的卻是 `-1`。

**判斷準則：含有輸入元件、且被區塊外按鈕使用的 Accordion 不可折疊。**
- `⚙️ 生成參數設定` → 其控制項是 `run_btn` 的 inputs，而 `run_btn` 在區塊外 → **必須 open=True**
- `🗂️ 產物管理`、`🏆 官網範例`、`💡 提示詞範本` → 只含按鈕，或輸入僅供同區塊按鈕使用
  （要按就得先展開）→ 可安全折疊

### 10.3 這是本專案第三個「靜默失效」
1. 參考圖被 autogrow API 丟棄（§6）
2. 輪詢逾時回傳 `None` → 前端顯示空白圖（§6.7 之後修正）
3. 折疊 Accordion 送出預設值（本節）

共同模式皆為「**UI 看起來正常、實際送出的資料不對**」。
**維運準則：前端改動後，必須到 ComfyUI `/history` 核對實際送出的 workflow，不可只看畫面。**

---

## 11. 【2026-09-23】改寫引擎改用本機 vLLM：快 8~10 倍且零額外資源

### 11.1 緣起
官方 PE 模型走 transformers batch-1（容器內無 vLLM，ARM64 裝不起來），
單次改寫 144~160 秒，切換 task 另需 95 秒，且需 19GB 常駐記憶體 + 36GB 磁碟。
本機已有 `qwen3.8`（`unsloth/Qwen3.8-27B-NVFP4`）以 vLLM 常駐於 8006，
具 prefix caching、MTP 推測解碼、fp8 KV cache —— 若可用，成本近乎為零。

### 11.2 我的兩個錯誤判斷
1. **誤以為 8006 是純文字模型**，僅憑容器啟動參數的模型名稱推斷，未實測。
   實測送入圖片後 1.6 秒正確回答「深綠色上衣」，**它具備視覺能力**。
2. **首次測試回傳空白**（259 秒、0 字），差點據此判定不合格。實際原因：
   - `max_tokens=4096` 被 thinking 階段吃光，`finish_reason` 未達 stop
   - 讀錯欄位名：此服務使用 `reasoning`，非 `reasoning_content`

   **關鍵設定：`chat_template_kwargs: {"enable_thinking": false}`。**

### 11.3 實測對照（官方 system prompt，官方 parse_answer 規則）
| 任務 | 本機 LLM (8006) | 官方 PE 模型 |
| :--- | :--- | :--- |
| t2i | **28.2 s**，`parse_ok: True`，`wh_ratio` 正常 | 159.7 s |
| edit | **17.6 s**，`parse_ok: True`，`ratio_follow: <image1>` | 144.3 s |
| 模型切換 | 無 | 95 s |
| 額外記憶體 | **0** | 19 GB |
| 額外磁碟 | **0** | 36 GB |

edit 任務的輸出品質同級：模型自行看圖辨識出原本是
「dark teal textured zip-up jacket」，並列出目標襯衫的尖領、前門襟、單邊胸袋、直筒長袖，
同時要求保留原棚拍光線與色調。

### 11.4 處置
`gradio_app.py` 新增 `call_llm()` 與「改寫引擎」下拉選單，**預設走本機 LLM**，
保留官方 PE 模型為可選後端（官方曾警告非微調模型未必穩定遵守輸出格式，
本處樣本僅兩筆，故不逕行移除）。前端實測：t2i 改寫 31.1 秒、`wh_ratio` 自動套用。

若後續使用穩定，可停用 PE 服務省 19GB 記憶體、刪除兩個 checkpoint 回收 36GB 磁碟。

### 11.5 版面：生成參數移至頁面最下方
使用者希望參數區不佔首屏，但 §10.2 已證實折疊會使其子元件卸載、送出預設值。
改為**保持展開但移到頁面最下方**，兼顧兩者。
端到端驗證（載入官方範例 ⑤ 後直接生成）：
`images=['image1','image2']`、`seed=1070478148268574`、`steps=40`、`cfg=1.0`、
`unet=int8_convrot` 全部正確送出。

---

## 12. 【2026-09-23】回看：生成紀錄，以及輸出檔名撞號的根治

### 12.1 為什麼需要自行記錄
ComfyUI 的 `/history` 只保留**實際送出**的提示詞，也就是擴寫後的版本。
使用者原本輸入的短句只存在於當時瀏覽器的輸入框，任何地方都沒有留存。
要做「原始 vs 擴寫」對照，必須由前端自行記錄。

新增 `generation_history.jsonl`（host 端 `qwen-image/` 下，容器重啟不受影響），
每次生成寫入一筆：時間、原始提示、擴寫提示、檔名、模式、尺寸、seed、步數、精度。
寫入失敗整段吞例外 —— 記錄功能不得影響出圖。

### 12.2 UI
頁面最下方「🕘 回看 (生成紀錄)」，表格四欄：時間（含模式/尺寸/seed 小字）、
原始提示、擴寫提示、縮圖。縮圖內嵌為 base64 data URI，
點擊透過 Gradio 檔案端點（`allowed_paths=[COMFY_OUTPUT_DIR]`）開新分頁看原圖，
瀏覽器不需直連 ComfyUI 的 8188。

### 12.3 回填舊紀錄，與暴露出的撞號問題
`scripts/backfill_history.py` 由 `/history` 回填既有生成（僅 `QwenImage21_*`，
排除開發測試輸出）。**原始提示無法回填**，該欄留空。

回填時抓到 22 筆但磁碟只有 11 張圖：**先前清空過 output 目錄，
ComfyUI 的流水號從 `00001` 重新開始，同一檔名被使用兩次。**
舊紀錄的 `file` 指向的檔案「存在」，但內容已是後來生成的另一張圖。

這個錯誤**無法在讀取端偵測**，回看表格會若無其事顯示錯誤的縮圖 ——
與 §10.3 所列的三個靜默失效同一性質。
已以「同檔名保留最新一筆」去重為 11 筆。

### 12.4 根治
`SaveImage` 的 `filename_prefix` 由固定的 `QwenImage21_Out` 改為
`time.strftime("QwenImage21_%Y%m%d_%H%M%S")`，檔名全域唯一，
清理輸出後也不會與舊檔撞號。

### 12.5 共用目錄的耦合
`COMFY_OUTPUT_DIR` 同時被四個功能存取：ComfyUI 寫入、打包下載讀取、
清理刪除、回看讀縮圖。檔案被清理時回看顯示「（檔案已清理）」而非破圖；
撞號問題則由 §12.4 的唯一檔名根治。

### 12.6 已知小瑕疵
回填紀錄使用主機本地時間，新紀錄使用容器內 UTC，兩者有時差。
不影響排序正確性，尚未統一。

---

## 13. 【2026-09-23】社群做法調查，與資源配置定案

### 13.1 網路上其他人怎麼做擴寫
官方建議用自家的 PE 模型，但社群實際上分三派：

| 做法 | 代表專案 | 記憶體 | 對齊官方 |
| :--- | :--- | ---: | :--- |
| 跑官方 PE checkpoint | [benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer](https://github.com/benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer) | ~20 GB (bf16) / ~10 GB (int8) | 完全 |
| 改用通用 Qwen LLM | [lihaoyun6/ComfyUI-QwenPromptRewriter](https://github.com/lihaoyun6/ComfyUI-QwenPromptRewriter) | 0（雲端 API） | 自訂 prompt |
| 泛用本機 LLM 節點 | [EricRollei/Local_LLM_Prompt_Enhancer](https://github.com/EricRollei/Local_LLM_Prompt_Enhancer)、[BigStationW/ComfyUI-Prompt-Rewriter](https://github.com/BigStationW/ComfyUI-Prompt-Rewriter/) | 視模型 | 不綁定 |
| **本部署：本機 vLLM + 官方 system prompt** | — | **0** | prompt 完全對齊，模型不同 |

兩個有價值的發現：
- 社群有提供**官方 PE 的 int8 量化版**（`qwen3.5_9b_qwen_image_2.1_pe_*.int8_convrot.safetensors`，
  各 9.47 GB），比我們下載的 bf16 原版（18 GB）小一半。若日後要回頭用官方 PE，應改用此版。
- benjiyaya 明載「VRAM: ~20GB for the PE model in bfloat16. A 24GB GPU is comfortable」——
  印證 §9.5 遇到的記憶體壓力並非本機個案。
- lihaoyun6 的做法與本部署同構（通用 Qwen LLM 擔任改寫器），但走阿里雲 API 且未公開其
  system prompt；本部署直接使用官方 repo 的 `system_prompt_t2i.txt` / `system_prompt_edit.txt`
  原檔，對齊程度更高。

### 13.2 三種「int8」不可混淆
| 對象 | 實際使用 |
| :--- | :--- |
| 影像模型（UNet + text encoder） | **int8_convrot**（§8 實測全勝） |
| 擴寫引擎 | **本機 8006 `qwen3.8-27B-NVFP4`** — 與 int8 無關 |
| 官方 PE checkpoint | **bf16 18GB × 2，閒置** |

### 13.3 PE 服務停用
改寫預設走 8006 之後，PE 服務屬於純閒置卻仍常駐 19GB。已停止：

| | 停止前 | 停止後 |
| :--- | ---: | ---: |
| 已用記憶體 | 82 GB | **63 GB** |
| 可用 | 38 GB | **58 GB** |

停止後實測改寫仍正常（22.3 秒，輸出含合格 `rewritten_prompt`）。

**PE 服務並未寫入任何自動啟動設定**（原本就是以 `docker exec -d` 手動啟動），
因此 ComfyUI 容器重啟後不會自行回來 —— 這是刻意的。需要時手動啟動：

```bash
docker exec -d qwen-image-comfyui sh -c   'cd /workspace/ComfyUI/pe && python3 pe_server.py > /workspace/ComfyUI/pe/pe_server.log 2>&1'
```

未啟動時於 WebUI 選「🎯 官方 PE 模型」會顯示明確的連線錯誤，不會靜默失敗。

### 13.4 磁碟上的 36GB PE checkpoint 暫不刪除
本機 LLM 切換僅一日，樣本不足；官方亦警告非微調模型未必穩定遵守輸出格式。
待累積使用、確認未出現 `parse_ok: false` 後再回收。磁碟尚有 61 GB，不急。

---

## 14. 【2026-09-23】改寫引擎雙端點分流，與遺漏的 presence_penalty

### 14.1 兩個本機端點，能力不同
| 端點 | 模型 | 視覺 | 文生圖改寫 | 備註 |
| :--- | :--- | :---: | ---: | :--- |
| **8002** | `Qwen3.6-35B-A3B-abliterated.i1-Q4_K_M.gguf`（llama.cpp） | ❌ | **約 5 秒** | 無 mmproj，送圖回 `HTTP 500: image input is not supported` |
| **8006** | `qwen3.8-27B-NVFP4`（vLLM） | ✅ | 22~31 秒 | 改圖改寫 17.6 秒 |

注意 8002 **並非**先前已停止的 `qwen3.6-nvfp4` 容器，是另一個 llama.cpp 服務，
且其 `/v1/models` 回傳 `{"models": [...]}` 而非 OpenAI 標準的 `{"data": [...]}`。

### 14.2 分流設計
`pick_endpoint(backend, task)` 依任務選擇端點，選單三項：
- `⚡ 自動`（預設）：文生圖 → 8002 取其速度；改圖／多圖 → 8006（唯一具視覺能力者）
- `8002` / `8006`：手動指定。選 8002 而任務需要視覺時，回傳明確提示而非送出去踩 500。

官方 PE 模型不列入選單（`call_pe` 與 `pe_server.py` 保留於程式中備用）。

### 14.3 遺漏的 presence_penalty
`call_llm()` 初版未帶 `presence_penalty`，而官方 `pe_core` profile 明定
**t2i 為 1.5、edit 為 0.0**。後果是輸出長度大幅不穩：

| 設定 | 耗時 | 擴寫長度 |
| :--- | ---: | ---: |
| 未帶（連測三次） | 2.3~4.4 s | **724 / 1073 / 1479** |
| 未關 thinking | 98.0 s | 3067 |
| **補上 1.5** | 5.3 s | **1625（穩定）** |

前端實測曾出現僅 103 字元的一句話擴寫，`finish_reason` 為 `stop`（非截斷），
即模型自行提前收尾。補上後問題消失。

**教訓**：改用非官方推論後端時，官方的**取樣參數**與 system prompt 同等重要，
不可只搬 prompt。
