# 🎨 Qwen-Image-2.1 DGX Spark 完整專案手冊

本專案封裝了部署於 **NVIDIA DGX Spark（GB10 Grace Blackwell ARM64 架構，<DGX_HOST>）** 上的 **Qwen-Image-2.1** 頂級影像生成與多模態智能編輯平台。

---

## 🚀 快速訪問與一鍵啟動

| 服務項目 | 連線網址 | 快速捷徑（點擊本機腳本） | 說明 |
| :--- | :--- | :--- | :--- |
| **Gradio 現代化 WebUI** | 👉 **[http://<DGX_HOST>:7860](http://<DGX_HOST>:7860)** | [`scripts/open_webui.bat`](file:///C:/Users/USER/qwen-image-dgx-spark/scripts/open_webui.bat) | 日常推薦使用。支援畫圈替換、遮罩重繪、多圖參考與剪貼簿快速貼上。 |
| **ComfyUI 專業工作台** | 👉 **[http://<DGX_HOST>:8188](http://<DGX_HOST>:8188)** | [`scripts/open_comfyui.bat`](file:///C:/Users/USER/qwen-image-dgx-spark/scripts/open_comfyui.bat) | 節點級工作流。支援高階工作流調整、節點串接與除錯。 |
| **SSH 終端連線** | `ssh <SSH_USER>@<DGX_HOST>` | [`scripts/ssh_connect.bat`](file:///C:/Users/USER/qwen-image-dgx-spark/scripts/ssh_connect.bat) | 直接登入 DGX Spark Linux 伺服器終端。 |

---

## 📂 專案目錄結構

```text
C:\Users\USER\qwen-image-dgx-spark\
├── README.md                      # 專案總導覽與快速啟動指南（本文件）
├── DEPLOYMENT_NOTES.md            # 完整技術部署筆記（架構、硬體、踩坑排查、官方技術規範）
├── FRONTEND_UPGRADE_PLAN.md       # 前端升級計畫書（MMDiT 視覺引導、筆刷遮罩、多圖參考）
├── app\
│   ├── gradio_app.py              # 前端 WebUI 完整原始碼（支援 Gradio 6 畫布、剪貼簿貼上、多圖管理）
│   ├── docker-compose.yml         # DGX Spark 雙容器編排配置檔（ComfyUI + Gradio）
│   └── qwen-image.service         # Linux systemd 開機自動啟動服務定義檔
└── scripts\
    ├── open_webui.bat             # 一鍵在瀏覽器開啟 Gradio WebUI (Port 7860)
    ├── open_comfyui.bat           # 一鍵在瀏覽器開啟 ComfyUI (Port 8188)
    ├── ssh_connect.bat            # 一鍵連線至 DGX Spark 伺服器
    └── sync_code_to_dgx.bat       # 一鍵將本機 gradio_app.py 同步至遠端並自動重啟容器
```

---

## 🌟 核心功能與操作技巧

### 1. 畫圈視覺引導替換（Circle Grounding）
- **操作方式**：選擇主要模式為「以圖改圖」，子模式選「🎯 畫圈視覺引導」。
- **畫布操作**：使用紅色筆刷在目標物體畫圈（可多色圈選：紅、綠、藍）。
- **提示詞寫法**：直接在 Prompt 中指定顏色，如：  
  `In <image1>, replace the object inside the red circle with a futuristic astronaut helmet, preserving facial likeness and lighting.`
- **底層技術**：帶圈影像直接傳遞給內建 **Qwen3-VL (8B)** 視覺大模型，大模型以空間注意力進行語義理解，Latent 採用乾淨原圖去噪，不殘留彩色墨水筆跡。

### 2. 遮罩局部精準重繪（Mask Inpainting）
- **操作方式**：子模式選「🔒 遮罩局部精準重繪」。
- **自動孔洞填滿防呆**：整合 `scipy.ndimage.binary_fill_holes`，使用者即使畫了空心圓圈，後端也會自動實心填滿內部，杜絕「雙重人臉鬼影」與「鋸齒孔洞」。
- **效果**：圈外像素 100% 嚴格凍結，圈內自動進行 6px 邊緣羽化融合。

### 3. 多圖參考庫（Multi-Reference Composition，最高 10 張）
- **操作方式**：在「多圖參考庫」紫色虛線框內，直接按 `Ctrl + V` 即可連續貼上圖片（自動編號為 `<image1>`、`<image2>`...）。
- **五圖合成穿搭範本**：  
  `A high-fashion full-body editorial portrait of the model from <image1>, wearing clothing from <image2>, shoes from <image3>, bag from <image4>, hat from <image5>, standing in a minimalist studio, sharp details, single image output.`

### 4. 剪貼簿貼上（Clipboard Paste）
- 瀏覽器端智能追蹤滑鼠位置（Hover-Aware Paste Routing）：
  - 滑鼠停在「主畫布區」按 `Ctrl + V` ➔ 貼入主畫布。
  - 滑鼠停在「參考圖區」按 `Ctrl + V` ➔ 依序追加至多圖庫。

---

## ⚙️ 官方黃金參數標準

已於 2026-09-22 逐項核對 [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) README 與 ComfyUI 官方模板
（`image_qwen_image_2_1_t2i` / `image_qwen_image_2_1_image_edit`）：

- **CFG 引導強度**：**`1.0`**（官方 `guidance_scale` 預設 1）。
- **負向提示詞**：**留空 `""`**（cfg=1 時不生效）。
- **取樣器**：**Euler + Simple**。
- **推論步數**：**40 步**（官方 README 所有範例皆 `num_inference_steps=40`）。
- **精度模式**：**bf16**（官方 `torch_dtype=torch.bfloat16`；UNETLoader `weight_dtype=default`）。
- **解析度**：**原生 2K**。官方 `aspect_ratios`：
  `1:1 2048x2048`、`4:3 2400x1792`、`3:4 1792x2400`、`3:2 2528x1696`、`2:3 1696x2528`、`16:9 2752x1536`、`9:16 1536x2752`。
- **改圖路徑**：latent 取 `TextEncodeQwenImage21` 的 output 2、`denoise=1.0`、`resolution=0`。
  官方是「整張重新生成」，不是局部重繪 — 要保留背景須在 prompt 明寫
  （例：`keep the original background and original lighting`）。

### ✅ 2026-09-22 重大修正：參考圖傳遞

ComfyUI 0.37.0 的 autogrow 動態輸入無法透過 `/prompt` API 接收參考圖，
導致「以圖改圖／多圖參考／畫圈引導」**從未真正運作**（只是純文字生圖）。
已新增 `ComfyUI/custom_nodes/qwen21_api_shim.py` 轉接節點解決，
內部直接呼叫官方 `TextEncodeQwenImage21.execute()`，不改演算法。
詳見 `DEPLOYMENT_NOTES.md` 第 6 節。

**自我檢查法**：改圖輸出尺寸若等於輸入圖尺寸 → 參考圖生效；
若恆為 1024x1024 正方形 → 參考圖被丟棄。

### ⚠️ 已知落差（尚未實作）

- **Prompt 改寫模型**：官方明示「For best results, we recommend using the official prompt rewriting models」，
  提供 `Qwen/Qwen-Image-2.1-PE-T2I` 與 `Qwen/Qwen-Image-2.1-PE-I2I`（Qwen3.5-VL 9B）。
  官網 showcase 的 prompt 幾乎都經過擴寫 — 未接此模型前，短 prompt 無法復現 showcase 品質。
- **遮罩作為標註**：官方 local editing 支援「circles, painted annotations, or separate masks」，
  但遮罩的角色是給 Qwen3-VL 看的 condition image，**不凍結像素**。
  舊版用 `VAEEncodeForInpaint` 凍結像素的做法非官方，已於本次移除。

### ❌ 本次修正的錯誤記載

先前本文件聲稱「官方標準」的下列數值，與官方 repo 不符，已更正：
`30 步` → 40；`fp8_e4m3fn_fast` → bf16；`1024 基準解析度` → 原生 2K。
fp8 會在載入時把官方 bf16 權重降級，是先前出圖品質不如官網的主因之一。

---

## 🛠️ 本機修改代碼並同步至 DGX

若您在本機修改了 [`app/gradio_app.py`](file:///C:/Users/USER/qwen-image-dgx-spark/app/gradio_app.py)，只需雙擊執行：
👉 [`scripts/sync_code_to_dgx.bat`](file:///C:/Users/USER/qwen-image-dgx-spark/scripts/sync_code_to_dgx.bat)

腳本會自動透過 SSH/SCP 將檔案上傳至遠端 `/home/<SSH_USER>/qwen-image/gradio_app.py`，並自動執行 `docker restart qwen-gradio-ui` 使變更立即生效！
