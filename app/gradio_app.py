import gradio as gr
import urllib.request
import urllib.error
import json
import io
import time
import os
import base64
import shutil
import tempfile
import zipfile
import numpy as np
import scipy.ndimage
from PIL import Image, ImageChops

COMFY_HOST = "127.0.0.1:8188"

# 模式識別碼。以穩定 id 取代先前的長字串比對 —— 改標題文字不會再牽動邏輯。
MODE_EDIT = "edit"
MODE_MULTI = "multi"
MODE_T2I = "t2i"

# 權重精度選項。int8_convrot 是官方 ComfyUI 模板的預設值：實測熱機推論快 14%、
# 冷載入快 2.25 倍、常駐記憶體省 14GB，文字渲染與身分保真皆無退步。
WEIGHT_PRESETS = {
    "int8 (官方模板預設，快且省記憶體)": (
        "qwen_image_2.1_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors"),
    "bf16 (最高保真，佔用約 30GB)": (
        "qwen_image_2.1_bf16.safetensors", "qwen3vl_8b_bf16.safetensors"),
}
DEFAULT_PRECISION = "int8 (官方模板預設，快且省記憶體)"
PE_HOST = "127.0.0.1:8200"   # 官方 Prompt Enhancer 常駐服務 (pe/pe_server.py)

# 官方 PE 回傳的 wh_ratio -> 本 UI 尺寸選項
PE_RATIO_TO_CHOICE = {
    "1:1": "1:1 (2048x2048) 官方預設",
    "4:3": "4:3 (2400x1792)",
    "3:4": "3:4 (1792x2400)",
    "3:2": "3:2 (2528x1696)",
    "2:3": "2:3 (1696x2528)",
    "16:9": "16:9 (2752x1536) 全景",
    "9:16": "9:16 (1536x2752)",
}


def pil_to_b64(pil_img):
    """PIL -> base64 PNG (PE 服務端會依官方 image_max_pixels 再縮放)"""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_pe(task, user_prompt, pil_images=None, seed=42, timeout=900):
    """呼叫官方 Prompt Enhancer。回傳 dict 或 raise。"""
    payload = {
        "task": task,
        "prompt": user_prompt,
        "images": [pil_to_b64(im) for im in (pil_images or [])],
        "seed": int(seed) if int(seed) != -1 else 42,
    }
    req = urllib.request.Request(
        f"http://{PE_HOST}/rewrite",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def upload_pil_to_comfy(pil_img, prefix="qwen_input"):
    """將 PIL 圖像上傳至 ComfyUI 的 input 目錄"""
    if pil_img is None:
        return None
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    img_byte_arr = io.BytesIO()
    
    # 遮罩或是具有透明度的圖使用 PNG 格式保存
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        pil_img.save(img_byte_arr, format="PNG")
    elif pil_img.mode == "L":
        pil_img.save(img_byte_arr, format="PNG")
    else:
        pil_img.convert("RGB").save(img_byte_arr, format="PNG")
        
    img_bytes = img_byte_arr.getvalue()
    filename = f"{prefix}_{int(time.time() * 1000)}.png"
    
    data = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    
    req = urllib.request.Request(
        f"http://{COMFY_HOST}/upload/image",
        data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res.get("name")

def parse_editor_data(editor_value, auto_fill_circles=True):
    """
    從 Gradio 6 的 gr.ImageEditor (type='pil') 解析背景圖、合成圖與遮罩
    若用戶畫的是空心圈，自動啟用形態學孔洞填充 (binary_fill_holes)，確保整塊物體實心被重繪，防止鋸齒邊界與雙重人臉
    """
    if not editor_value:
        return None, None, None, False
        
    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        composite = editor_value.get("composite")
        layers = editor_value.get("layers", [])
    else:
        return editor_value.convert("RGB"), editor_value.convert("RGB"), None, False

    if bg is None and composite is not None:
        bg = composite.convert("RGB")
    elif bg is not None:
        bg = bg.convert("RGB")
        
    if composite is not None:
        composite = composite.convert("RGB")
    else:
        composite = bg

    has_mask = False
    combined_mask = None
    
    if layers and len(layers) > 0:
        for layer in layers:
            if layer is not None:
                layer_rgba = layer.convert("RGBA")
                alpha = layer_rgba.split()[-1]
                extrema = alpha.getextrema()
                if extrema and extrema[1] > 10:
                    has_mask = True
                    binary_mask = alpha.point(lambda p: 255 if p > 10 else 0).convert("L")
                    if combined_mask is None:
                        combined_mask = binary_mask
                    else:
                        combined_mask = ImageChops.lighter(combined_mask, binary_mask)
                        
    if has_mask and combined_mask is not None and auto_fill_circles:
        # 自動閉合與填滿空心圓圈內部
        arr = np.array(combined_mask) > 128
        # 先進行 5px 膨脹閉合微小斷縫，再填孔
        struct = scipy.ndimage.generate_binary_structure(2, 2)
        closed = scipy.ndimage.binary_closing(arr, structure=struct, iterations=3)
        filled = scipy.ndimage.binary_fill_holes(closed)
        combined_mask = Image.fromarray((filled * 255).astype(np.uint8))
        
    return bg, composite, combined_mask, has_mask

def send_to_comfy(
    mode, 
    edit_submode, 
    prompt, 
    negative_prompt, 
    editor_input, 
    ref_images_list, 
    aspect_ratio, 
    steps, 
    cfg, 
    seed,
    precision=DEFAULT_PRECISION
):
    ref_images_list = ref_images_list or []
    unet_name, clip_name = WEIGHT_PRESETS.get(precision, WEIGHT_PRESETS[DEFAULT_PRECISION])
    
    # 基礎模型架構 (對齊官方模板: weight_dtype=default + QwenImage21Cache，精度由 UI 選擇)
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "qwen_image"
            }
        },
        "9": {
            "class_type": "QwenImage21Cache",
            "inputs": {
                "model": ["1", 0],
                "device": "auto",
                "dtype": "default"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_2.1_vae_bf16.safetensors"
            }
        }
    }

    images_dict = {}
    node_counter = 100

    # 模式判斷
    if mode == MODE_EDIT:
        bg_img, composite_img, mask_img, has_user_drawn = parse_editor_data(editor_input, auto_fill_circles=True)
        if bg_img is None and len(ref_images_list) > 0:
            bg_img = ref_images_list[0]
            
        if bg_img is not None:
            # 1. 畫圈視覺引導 (官方 image_edit 路徑，image_1 為使用者標註過的圖)
            if (edit_submode == "🎯 畫圈視覺引導 (Circle Grounding - 帶彩色圈交給 Qwen3-VL 識別)") and has_user_drawn:
                # 傳遞帶手繪彩色圈的圖像給 Qwen3-VL 作為視覺引導提示
                comp_name = upload_pil_to_comfy(composite_img, prefix="circled_img")
                workflow["10"] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": comp_name}
                }
                images_dict["image1"] = ["10", 0]
                # 官方 image_edit 模板：latent 取 TextEncodeQwenImage21 的 output 2 (依 image_1 尺寸的空 latent)
                latent_source = ["4", 2]
                actual_denoise = 1.0
                
            # 2. 一般全圖改圖 (官方 image_edit 路徑)
            else:
                bg_name = upload_pil_to_comfy(bg_img, prefix="global_img")
                workflow["10"] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": bg_name}
                }
                images_dict["image1"] = ["10", 0]
                latent_source = ["4", 2]
                actual_denoise = 1.0

            # 額外參考圖注入為 image_2, image_3 ...
            for idx, extra_img in enumerate(ref_images_list, 2):
                extra_name = upload_pil_to_comfy(extra_img, prefix=f"ref_{idx}")
                nid = str(node_counter)
                node_counter += 1
                workflow[nid] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": extra_name}
                }
                images_dict[f"image{idx}"] = [nid, 0]

            images_kwargs = dict(images_dict)
            workflow["4"] = {
                "class_type": "TextEncodeQwenImage21API",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "prompt": prompt,
                    "negative_prompt": negative_prompt if negative_prompt else "",
                    "resolution": 1024,
                    **images_kwargs
                }
            }
        else:
            mode = MODE_T2I

    elif mode == MODE_MULTI:
        if len(ref_images_list) > 0:
            for idx, ref_img in enumerate(ref_images_list, 1):
                ref_name = upload_pil_to_comfy(ref_img, prefix=f"ref_{idx}")
                nid = str(node_counter)
                node_counter += 1
                workflow[nid] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": ref_name}
                }
                images_dict[f"image{idx}"] = [nid, 0]

            images_kwargs = dict(images_dict)
            workflow["4"] = {
                "class_type": "TextEncodeQwenImage21API",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "prompt": prompt,
                    "negative_prompt": negative_prompt if negative_prompt else "",
                    "resolution": 1024,
                    **images_kwargs
                }
            }
            # 官方標準：直接調用 TextEncodeQwenImage21 的輸出埠 2 (乾淨空畫布)
            latent_source = ["4", 2]
            actual_denoise = 1.0
        else:
            mode = MODE_T2I

    if mode == MODE_T2I:
        # 官方 README aspect_ratios (原生 2K，皆為 32 倍數)
        size_map = {
            "1:1 (2048x2048) 官方預設": (2048, 2048),
            "4:3 (2400x1792)": (2400, 1792),
            "3:4 (1792x2400)": (1792, 2400),
            "3:2 (2528x1696)": (2528, 1696),
            "2:3 (1696x2528)": (1696, 2528),
            "16:9 (2752x1536) 全景": (2752, 1536),
            "9:16 (1536x2752)": (1536, 2752),
            "1:1 省時 (1024x1024)": (1024, 1024)
        }
        w, h = size_map.get(aspect_ratio, (2048, 2048))
        workflow["4"] = {
            "class_type": "TextEncodeQwenImage21API",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": prompt,
                "negative_prompt": negative_prompt if negative_prompt else "",
                "resolution": 1024
            }
        }
        workflow["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": w,
                "height": h,
                "batch_size": 1
            }
        }
        latent_source = ["5", 0]
        actual_denoise = 1.0

    # 取樣器 (完全對齊官方標準：Euler + Simple，CFG 預設 1.0)
    workflow["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["9", 0],
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": latent_source,
            "seed": int(seed) if seed != -1 else int(time.time() * 1000) % 1000000007,
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": actual_denoise
        }
    }
    workflow["7"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["6", 0],
            "vae": ["3", 0]
        }
    }
    workflow["8"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "QwenImage21_Out",
            "images": ["7", 0]
        }
    }

    req_data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=req_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        prompt_id = res.get("prompt_id")
    
    # 輪詢等待推論完成。
    # 2048x2048 / 40 步冷啟動實測約 220 秒，再加上佇列等待，逾時上限設為 30 分鐘。
    # 舊版為 180 秒且逾時後靜默回傳 None，會讓「其實已生成成功」的任務在前端顯示空白。
    POLL_TIMEOUT_S = 1800
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(2)
        with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as h_resp:
            h_data = json.loads(h_resp.read().decode("utf-8"))
        if prompt_id not in h_data:
            continue
        entry = h_data[prompt_id]
        outputs = entry.get("outputs", {})
        if "8" in outputs and "images" in outputs["8"]:
            img_info = outputs["8"]["images"][0]
            img_url = (f"http://{COMFY_HOST}/view?filename={img_info['filename']}"
                       f"&subfolder={img_info['subfolder']}&type={img_info['type']}")
            with urllib.request.urlopen(img_url) as img_resp:
                return Image.open(io.BytesIO(img_resp.read()))
        status = entry.get("status", {})
        if status.get("status_str") and status.get("status_str") != "success":
            msgs = json.dumps(status.get("messages", [])[-2:], ensure_ascii=False)[:500]
            raise gr.Error(f"ComfyUI 執行失敗：{status.get('status_str')} / {msgs}")

    raise gr.Error(
        f"等待逾時（{POLL_TIMEOUT_S} 秒）。任務可能仍在 ComfyUI 佇列中執行，"
        f"prompt_id={prompt_id}，可至 http://{COMFY_HOST} 查看結果。")


# =============================================================================
# 官網範例復現資料表
# 來源：https://github.com/QwenLM/Qwen-Image-2.1 README 的程式碼範例，
#       以及 ComfyUI 官方模板 image_qwen_image_2_1_image_edit 的輸入素材。
# 官方所有範例一律 num_inference_steps=40、seed=42、guidance_scale=1、negative 留空。
# =============================================================================
OFFICIAL_SAMPLES_DIR = "/workspace/official_samples"

OFFICIAL_DEMOS = {
    "t2i_neon": {
        "label": "① 文生圖：霓虹招牌",
        "mode": MODE_T2I,
        "prompt": 'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
        "ratio": "1:1 (2048x2048) 官方預設",
        "seed": 42,
        "images": [],
        "tip": "官方 README 第一個範例。重點在文字渲染是否正確拼出 QWEN IMAGE 2.1。",
    },
    "t2i_rgba": {
        "label": "② 透明圖 RGBA：卡通龍貼紙",
        "mode": MODE_T2I,
        "prompt": "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.",
        "ratio": "1:1 (2048x2048) 官方預設",
        "seed": 42,
        "images": [],
        "tip": "原生透明通道。輸出區的棋盤格底代表透明；請另存 PNG 才會保留 alpha。",
    },
    "t2i_pano": {
        "label": "③ 全景 16:9",
        "mode": MODE_T2I,
        "prompt": "A panoramic mountain landscape",
        "ratio": "16:9 (2752x1536) 全景",
        "seed": 42,
        "images": [],
        "tip": "官方 aspect_ratios 的 16:9 = 2752x1536，這是原生 2K 級全景。",
    },
    "edit_bg": {
        "label": "④ 單圖編輯：換成夕陽海灘",
        "mode": MODE_EDIT,
        "prompt": "Change the background to a sunset beach",
        "ratio": None,
        "seed": 42,
        "images": ["portrait_model_denim.png"],
        "tip": "官方 README 的單圖編輯範例。注意官方是整張重生成，人物應保留、背景改變。",
    },
    "edit_denim": {
        "label": "⑤ 換衣（官方模板範例）",
        "mode": MODE_EDIT,
        "prompt": "Keep the character and pose in <image1> unchanged, put this light blue denim shirt from <image2> on the character, preserve the original facial features, hair, body shape and pose, the denim shirt fits naturally on body, realistic denim fabric texture, natural clothing folds, keep the original background and original lighting, high fashion editorial photography, sharp details",
        "ratio": None,
        "seed": 1070478148268574,
        "images": ["portrait_model_denim.png", "clothing_light_blue_denim_shirt.png"],
        "tip": "ComfyUI 官方模板附的完整範例，連 seed 都是官方值。最適合用來判斷是否對齊。",
    },
}


def load_official_demo(key):
    """一鍵載入官網範例：模式、提示詞、尺寸、seed、輸入圖全部設為官方值。"""
    d = OFFICIAL_DEMOS[key]
    paths = [os.path.join(OFFICIAL_SAMPLES_DIR, f) for f in d["images"]]
    imgs = []
    for path in paths:
        if os.path.exists(path):
            imgs.append(Image.open(path).convert("RGB"))

    main_img = imgs[0] if imgs else None
    refs = imgs[1:] if len(imgs) > 1 else []

    editor_val = {"background": main_img, "layers": [], "composite": main_img} if main_img else None
    ratio_val = gr.update(value=d["ratio"]) if d["ratio"] else gr.update()

    missing = [f for f, p in zip(d["images"], paths) if not os.path.exists(p)]
    note = f"　⚠️ 缺少素材：{', '.join(missing)}" if missing else ""

    return (
        d["mode"],                       # gr.State：模式識別碼
        gr.Tabs(selected=d["mode"]),     # 切換到對應分頁
        d["prompt"],
        ratio_val,
        gr.update(value=40),      # 官方 num_inference_steps
        gr.update(value=1.0),     # 官方 guidance_scale
        gr.update(value=d["seed"]),
        editor_val,
        refs,
        refs,
        gr.update(visible=(d["mode"] == MODE_EDIT)),
        gr.update(visible=(d["mode"] != MODE_T2I)),
        f"📋 已載入「{d['label']}」：{d['tip']}{note}",
    )



# =============================================================================
# 產物管理：打包下載 / 清理
# Gradio 容器掛載 /home/<user>/qwen-image -> /workspace，故可直接存取 ComfyUI 目錄。
# =============================================================================
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output")
COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", "/workspace/ComfyUI/input")


def _human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}" if u != "B" else f"{int(n)}B"
        n /= 1024.0


def _collect(directory, days):
    """回傳 (檔案清單, 總位元組)。days<=0 表示不限時間，全部納入。"""
    if not os.path.isdir(directory):
        return [], 0
    cutoff = time.time() - days * 86400 if days > 0 else None
    files, total = [], 0
    for name in os.listdir(directory):
        fp = os.path.join(directory, name)
        if not os.path.isfile(fp):
            continue
        st = os.stat(fp)
        if cutoff is not None and st.st_mtime < cutoff:
            continue
        files.append(fp)
        total += st.st_size
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files, total


def pack_outputs(days):
    """把輸出圖片打包成 zip 供下載。days=0 表示全部。"""
    days = int(days)
    files, total = _collect(COMFY_OUTPUT_DIR, days)
    if not files:
        scope = "全部" if days <= 0 else f"最近 {days} 天"
        return None, f"⚠️ {scope}沒有可打包的圖片。"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(tempfile.gettempdir(), f"qwen_outputs_{stamp}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # PNG 已壓縮，不再重壓
        for fp in files:
            z.write(fp, arcname=os.path.basename(fp))
    zsize = os.path.getsize(zip_path)
    scope = "全部" if days <= 0 else f"最近 {days} 天"
    return zip_path, f"✅ 已打包{scope} **{len(files)} 張**圖片（原始 {_human(total)}，壓縮檔 {_human(zsize)}）。點下方檔案下載。"


def _delete(files):
    """刪除檔案清單，回傳 (成功數, 失敗數, 釋放位元組)。"""
    ok = errs = freed = 0
    for f in files:
        try:
            sz = os.path.getsize(f)
            os.remove(f)
            ok += 1
            freed += sz
        except OSError:
            errs += 1
    return ok, errs, freed


#: 這些是 ComfyUI 自帶的佔位檔，任何情況都不刪
PROTECTED_NAMES = {"example.png", "_output_images_will_be_put_here",
                   "put_input_images_here", ".keep", ".gitkeep"}


def _purgeable(directory):
    """回傳該目錄可刪除的檔案，新到舊排序。"""
    files, _ = _collect(directory, 0)
    return [f for f in files if os.path.basename(f) not in PROTECTED_NAMES]


def cleanup_keep_recent(keep_out, keep_in, confirm):
    """只保留最近 N 張，其餘刪除。N=0 表示該項不處理。"""
    keep_out, keep_in = int(keep_out), int(keep_in)
    lines = []
    for label, directory, keep in (
        ("輸出圖片", COMFY_OUTPUT_DIR, keep_out),
        ("上傳暫存", COMFY_INPUT_DIR, keep_in),
    ):
        files = _purgeable(directory)
        if keep <= 0:
            lines.append(f"- **{label}**：共 {len(files)} 張 — 保留張數設為 0，略過")
            continue
        victims = files[keep:]          # _collect 已由新到舊排序
        if not victims:
            lines.append(f"- **{label}**：共 {len(files)} 張，未超過保留上限 {keep} 張，無需清理")
            continue
        size = sum(os.path.getsize(f) for f in victims)
        if confirm:
            ok, errs, freed = _delete(victims)
            lines.append(f"- **{label}**：已刪除 **{ok}** 張較舊的，保留最近 {keep} 張，釋放 **{_human(freed)}**"
                         + (f"（{errs} 個失敗）" if errs else ""))
        else:
            lines.append(f"- **{label}**：共 {len(files)} 張，將刪除較舊的 **{len(victims)}** 張、"
                         f"保留最近 {keep} 張，可釋放 **{_human(size)}**")

    head = "🗑️ **已執行清理**" if confirm else "🔍 **預演結果（未刪除任何檔案）**"
    tail = "" if confirm else "\n\n勾選「我確認刪除」後再按一次，才會真的刪除。"
    return head + "\n" + "\n".join(lines) + tail


def purge_all(confirm_text):
    """清空全部產物。必須輸入 DELETE 才執行，避免誤觸。"""
    out_files = _purgeable(COMFY_OUTPUT_DIR)
    in_files = _purgeable(COMFY_INPUT_DIR)
    size = sum(os.path.getsize(f) for f in out_files + in_files)
    if (confirm_text or "").strip().upper() != "DELETE":
        return (f"⚠️ **尚未執行。** 將清空輸出 **{len(out_files)}** 張 + 上傳暫存 **{len(in_files)}** 個，"
                f"合計 **{_human(size)}**。\n\n確定要全部刪除，請在欄位輸入 `DELETE` 後再按一次。")
    ok1, e1, f1 = _delete(out_files)
    ok2, e2, f2 = _delete(in_files)
    errs = e1 + e2
    return (f"🗑️ **已清空。** 輸出刪除 **{ok1}** 張、上傳暫存刪除 **{ok2}** 個，"
            f"釋放 **{_human(f1 + f2)}**" + (f"（{errs} 個失敗）" if errs else ""))


custom_css = '''
.output-checkerboard {
    background-image: linear-gradient(45deg, #ddd 25%, transparent 25%), 
                      linear-gradient(-45deg, #ddd 25%, transparent 25%), 
                      linear-gradient(45deg, transparent 75%, #ddd 75%), 
                      linear-gradient(-45deg, transparent 75%, #ddd 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
}
#ref_paste_box {
    border: 2px dashed #4f46e5 !important;
    background-color: #f5f3ff !important;
}
'''

with gr.Blocks(title="Qwen-Image-2.1 官方標準工作站 (DGX Spark)", css=custom_css) as demo:
    ref_images_state = gr.State([])
    
    gr.Markdown("# 🎨 Qwen-Image-2.1 影像生成與智能圈選編輯平台 (官方規範版)")
    gr.Markdown("**硬體**: NVIDIA DGX Spark (GB10 Grace Blackwell ARM64) | **標準**: Single-Stream MMDiT + Qwen3-VL | **官方標準**: CFG 1.0 | Euler + Simple | 支援 1~10 張多圖參考")
    
    with gr.Accordion("🗂️ 產物管理 (打包下載 / 清理)", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**📦 打包下載**　把伺服器上的輸出圖片打包成 zip")
                pack_days = gr.Number(value=7, precision=0, label="打包最近幾天 (0 = 全部)")
                btn_pack = gr.Button("📦 打包下載", variant="secondary")
                pack_status = gr.Markdown("")
                pack_file = gr.File(label="下載 zip", interactive=False)
            with gr.Column():
                gr.Markdown("**🗑️ 清理**　保留最近 N 張，其餘刪除。預設只預演，勾選確認才會刪除")
                with gr.Row():
                    keep_out = gr.Number(value=100, precision=0, label="輸出圖片保留最近幾張 (0=不刪)")
                    keep_in = gr.Number(value=20, precision=0, label="上傳暫存保留最近幾個 (0=不刪)")
                confirm_del = gr.Checkbox(value=False, label="⚠️ 我確認刪除（不勾選只做預演）")
                btn_clean = gr.Button("🗑️ 執行清理", variant="stop")
                clean_status = gr.Markdown("")

                with gr.Accordion("💣 清空全部（危險）", open=False):
                    gr.Markdown("刪除**所有**輸出圖片與上傳暫存，不可復原。請先按一次查看數量。")
                    purge_text = gr.Textbox(label="輸入 DELETE 以確認", placeholder="DELETE", lines=1)
                    btn_purge = gr.Button("💣 清空全部", variant="stop")
                    purge_status = gr.Markdown("")

    with gr.Accordion("🏆 官網範例一鍵復現 (Reproduce Official Demos)", open=False):
        gr.Markdown(
            "點擊後會自動套用**官方原始的提示詞、尺寸、步數 40、CFG 1.0、seed 與輸入圖**，"
            "接著直接按 🚀 生成即可。用來判斷本機部署是否已對齊官方表現。"
        )
        with gr.Row():
            btn_demo_neon = gr.Button(OFFICIAL_DEMOS["t2i_neon"]["label"], size="sm")
            btn_demo_rgba = gr.Button(OFFICIAL_DEMOS["t2i_rgba"]["label"], size="sm")
            btn_demo_pano = gr.Button(OFFICIAL_DEMOS["t2i_pano"]["label"], size="sm")
            btn_demo_bg = gr.Button(OFFICIAL_DEMOS["edit_bg"]["label"], size="sm")
            btn_demo_denim = gr.Button(OFFICIAL_DEMOS["edit_denim"]["label"], size="sm", variant="primary")
        demo_status = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=5):
            # 分頁僅作為模式選擇器；畫布與參考圖庫為三個模式共用，
            # 故留在分頁之外，靠既有的顯示/隱藏邏輯切換，避免重複元件與狀態同步問題。
            mode = gr.State(MODE_EDIT)
            with gr.Tabs() as mode_tabs:
                with gr.TabItem("🎯 以圖改圖", id=MODE_EDIT):
                    gr.Markdown("上傳主圖後下指令。可用筆刷圈選要修改的位置，並加掛最多 10 張參考圖。")
                with gr.TabItem("🖼️ 多圖參考融合", id=MODE_MULTI):
                    gr.Markdown("貼上多張參考圖，用 `<image1>`、`<image2>` … 在提示詞中指名合成。")
                with gr.TabItem("✨ 純文字生圖", id=MODE_T2I):
                    gr.Markdown("不需圖片，直接輸入提示詞。尺寸由右側「生成參數」決定。")
            
            # --- 模式 1 容器: 畫布塗抹/圈選 ---
            with gr.Group(visible=True) as edit_canvas_container:
                edit_submode = gr.Radio(
                    choices=[
                        "🎯 畫圈視覺引導 (Circle Grounding - 帶彩色圈交給 Qwen3-VL 識別)",
                        "🖼️ 全圖改圖 (Global Edit - 直接送原圖)"
                    ],
                    value="🖼️ 全圖改圖 (Global Edit - 直接送原圖)",
                    label="🖌️ 編輯互動方式",
                    info="⚠️ 官方 Qwen-Image-2.1 無 inpaint 路徑：兩種方式都是整張重新生成 (denoise=1.0)。要保留背景請在 prompt 明寫 keep the original background and lighting unchanged。"
                )
                
                editor_input = gr.ImageEditor(
                    label="🖼️ 主圖 (畫圈視覺引導時可用彩色筆刷標註；點此按 Ctrl+V 可貼上)",
                    type="pil",
                    image_mode="RGBA",
                    sources=["upload", "clipboard"],
                    brush=gr.Brush(colors=["#ff0000", "#00ff00", "#0000ff", "#ffffff"], default_color="#ff0000", default_size=20),
                    eraser=gr.Eraser(),
                    elem_id="main_canvas_box"
                )
                
            # --- 模式 1 & 2 共用: 多張參考圖管理庫 (支援最多 10 張) ---
            with gr.Group(visible=True) as ref_images_container:
                gr.Markdown("### 📎 多圖參考庫 (支援 1~10 張，可不斷按 Ctrl+V 連續貼上新圖)")
                
                with gr.Row():
                    ref_paste_input = gr.Image(
                        label="📋 點此處按 Ctrl+V 貼上圖片 (貼上後自動新增至右方圖庫)",
                        type="pil",
                        sources=["upload", "clipboard"],
                        elem_id="ref_paste_box"
                    )
                    with gr.Column():
                        ref_files_input = gr.File(
                            label="📁 或拖曳/多選上傳多張圖",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        with gr.Row():
                            btn_clear_refs = gr.Button("🗑️ 清空所有參考圖", size="sm", variant="secondary")
                            btn_pop_ref = gr.Button("⏪ 移除最後一張", size="sm")

                ref_gallery = gr.Gallery(
                    label="🖼️ 當前已載入的參考圖集 (依序對應 <image1>, <image2>, ...)",
                    type="pil",
                    columns=4,
                    height="auto",
                    show_label=True
                )

            prompt = gr.Textbox(
                label="✏️ 修改指令 / 提示詞 (Prompt - 建議依官方規範使用 <image1>, <image2> 錨定)",
                placeholder="例如：A high-fashion full-body portrait of the model from <image1>, wearing clothing from <image2>...",
                lines=3
            )
            
            with gr.Row():
                btn_pe = gr.Button("✨ 官方 Prompt 改寫 (PE)", variant="secondary", size="lg")
                pe_status = gr.Markdown("")

            with gr.Accordion("💡 官方 GitHub 標準提示詞範本 (一鍵填入)", open=False):
                gr.Markdown("*遵循官方 `system_prompt_edit.txt` 規範：使用 `<imageX>` 明確指定主體與部件，並加入光影與細節保持約束。*")
                with gr.Row():
                    btn_tpl_outfit5 = gr.Button("五圖合成穿搭寫真", size="sm")
                    btn_tpl_circles2 = gr.Button("紅圈頭盔 + 綠圈髮箍", size="sm")
                    btn_tpl_swap_garment = gr.Button("單件衣服換穿 (<image2>)", size="sm")
                    btn_tpl_scene_merge = gr.Button("主體融入背景 (<image2>)", size="sm")

            neg_prompt = gr.Textbox(
                label="負向提示詞 (Negative Prompt - 官方標準推薦留空)",
                value="",
                placeholder="官方推薦留空以獲得最佳多圖融合品質",
                lines=1
            )
            
            run_btn = gr.Button("🚀 開始執行生成 / 替換 (Execute)", variant="primary", size="lg")

        with gr.Column(scale=4):
            # ⚠️ 必須保持展開：Gradio 6 的折疊 Accordion 不渲染子元件，
            # 提交時會送出預設值而非畫面上的值（實測 seed 顯示官方值、送出卻是 -1）。
            # 這些控制項是 run_btn 的 inputs，run_btn 在本區塊之外，故不可折疊。
            with gr.Accordion("⚙️ 生成參數設定 (2K / 40 步 / CFG 1.0 / int8，已對齊官方)", open=True):
                aspect_ratio = gr.Dropdown(
                    choices=[
                        "1:1 (2048x2048) 官方預設",
                        "4:3 (2400x1792)",
                        "3:4 (1792x2400)",
                        "3:2 (2528x1696)",
                        "2:3 (1696x2528)",
                        "16:9 (2752x1536) 全景",
                        "9:16 (1536x2752)",
                        "1:1 省時 (1024x1024)"
                    ],
                    value="1:1 (2048x2048) 官方預設",
                    label="文生圖尺寸 (官方 README aspect_ratios；以圖改圖時自動繼承原圖尺寸)"
                )
                precision = gr.Dropdown(
                    choices=list(WEIGHT_PRESETS.keys()),
                    value=DEFAULT_PRECISION,
                    label="⚙️ 權重精度",
                    info="切換精度會觸發模型重新載入（int8 約 18 秒、bf16 約 99 秒）"
                )
                steps = gr.Slider(minimum=15, maximum=50, value=40, step=1, label="推論步數 (官方 README: num_inference_steps=40)")
                cfg = gr.Slider(
                    minimum=1.0, maximum=5.0, value=1.0, step=0.1,
                    label="提示詞引導強度 (CFG)",
                    info="【官方標準：1.0】Diffusers 與官方 ComfyUI 預設值。設為 1.0 時無負向引導干擾。"
                )
                seed = gr.Number(value=-1, label="隨機種子 (Seed, -1 為隨機)", precision=0)
            
            output_img = gr.Image(
                label="🖼️ 輸出結果 (新圖像)",
                type="pil",
                image_mode="RGBA",
                elem_classes=["output-checkerboard"]
            )

    # --- 參考圖動態管理邏輯 ---
    def add_single_ref(new_img, current_list):
        current_list = list(current_list or [])
        if new_img is not None:
            if len(current_list) >= 10:
                gr.Warning("最多支援 10 張參考圖！已達到上限。")
                return [(img, f"<image{i+1}>") for i, img in enumerate(current_list)], current_list, None
            current_list.append(new_img.convert("RGB"))
        gallery_items = [(img, f"<image{i+1}>") for i, img in enumerate(current_list)]
        return gallery_items, current_list, None

    def add_batch_files(files, current_list):
        current_list = list(current_list or [])
        if files:
            for f in files:
                if len(current_list) >= 10:
                    break
                try:
                    fpath = f.name if hasattr(f, 'name') else f
                    img = Image.open(fpath).convert("RGB")
                    current_list.append(img)
                except Exception as e:
                    print("讀取檔案失敗:", e)
        gallery_items = [(img, f"<image{i+1}>") for i, img in enumerate(current_list)]
        return gallery_items, current_list, None

    def clear_all_refs():
        return [], [], None

    def remove_last_ref(current_list):
        current_list = list(current_list or [])
        if len(current_list) > 0:
            current_list.pop()
        gallery_items = [(img, f"<image{i+1}>") for i, img in enumerate(current_list)]
        return gallery_items, current_list, None

    ref_paste_input.change(
        fn=add_single_ref,
        inputs=[ref_paste_input, ref_images_state],
        outputs=[ref_gallery, ref_images_state, ref_paste_input]
    )
    
    ref_files_input.change(
        fn=add_batch_files,
        inputs=[ref_files_input, ref_images_state],
        outputs=[ref_gallery, ref_images_state, ref_files_input]
    )

    btn_clear_refs.click(
        fn=clear_all_refs,
        inputs=[],
        outputs=[ref_gallery, ref_images_state, ref_paste_input],
        queue=False
    )

    btn_pop_ref.click(
        fn=remove_last_ref,
        inputs=[ref_images_state],
        outputs=[ref_gallery, ref_images_state, ref_paste_input],
        queue=False
    )

    # 官方標準提示詞快捷填入 (基於 system_prompt_edit.txt)
    btn_tpl_outfit5.click(
        lambda: "A single full-body fashion editorial photograph of the model shown in <image1>, wearing the clothing from <image2>, wearing the shoes from <image3>, carrying the bag from <image4>, and wearing the hat from <image5>. The model stands in a bright minimalist photography studio. Natural soft lighting, realistic fabric folds, 8k resolution, elegant composition, single image output.",
        outputs=prompt,
        queue=False
    )
    btn_tpl_circles2.click(
        lambda: "In <image1>, replace the headwear inside the red circle with a futuristic white astronaut helmet featuring a transparent reflective visor, and replace the headwear inside the green circle with a cute alien antenna headband, preserving original background, lighting, facial expressions and identities.",
        outputs=prompt,
        queue=False
    )
    btn_tpl_swap_garment.click(
        lambda: "Keep the character and pose in <image1> strictly unchanged, put the clothing item shown in <image2> onto the character, preserve original facial likeness, body shape, skin texture and background lighting.",
        outputs=prompt,
        queue=False
    )
    btn_tpl_scene_merge.click(
        lambda: "Place the subject from <image1> into the environment of <image2>, matching the scene perspective, ambient illumination, cast shadows and depth of field, photorealistic 8k.",
        outputs=prompt,
        queue=False
    )

    # 模式切換動態顯示/隱藏容器
    def on_mode_change(m):
        is_canvas = (m == MODE_EDIT)
        is_multi = (m == MODE_MULTI)
        return gr.update(visible=is_canvas), gr.update(visible=(is_canvas or is_multi))

    def on_tab_select(evt: gr.SelectData):
        m = evt.value if isinstance(evt.value, str) else MODE_EDIT
        # Gradio 回傳的是頁籤標題，對照回識別碼
        m = {"🎯 以圖改圖": MODE_EDIT,
             "🖼️ 多圖參考融合": MODE_MULTI,
             "✨ 純文字生圖": MODE_T2I}.get(m, MODE_EDIT)
        vis_canvas, vis_refs = on_mode_change(m)
        return m, vis_canvas, vis_refs

    mode_tabs.select(
        fn=on_tab_select,
        inputs=None,
        outputs=[mode, edit_canvas_container, ref_images_container],
        queue=False          # 純前端顯示切換，不可排在生圖長任務後面
    )

    # 官網範例一鍵復現
    _demo_outputs = [
        mode, mode_tabs, prompt, aspect_ratio, steps, cfg, seed,
        editor_input, ref_images_state, ref_gallery,
        edit_canvas_container, ref_images_container, demo_status,
    ]
    for _btn, _key in (
        (btn_demo_neon, "t2i_neon"),
        (btn_demo_rgba, "t2i_rgba"),
        (btn_demo_pano, "t2i_pano"),
        (btn_demo_bg, "edit_bg"),
        (btn_demo_denim, "edit_denim"),
    ):
        _btn.click(
            fn=(lambda k=_key: load_official_demo(k)),
            inputs=[],
            outputs=_demo_outputs,
            queue=False
        )

    def enhance_prompt(m, p_text, editor_data, refs, sd):
        """官方 Prompt Enhancer：短 prompt -> 擴寫 prompt，並套用官方建議畫布比例。"""
        if not (p_text or "").strip():
            return gr.update(), gr.update(), "⚠️ 請先輸入提示詞再改寫。"

        refs = refs or []
        if m == MODE_T2I:
            task, imgs = "t2i", []
        else:
            task = "edit"
            imgs = []
            bg_img, composite_img, _mask, has_drawn = parse_editor_data(
                editor_data, auto_fill_circles=False)
            # 官方 local editing 明示支援 circles / painted annotations：
            # 有手繪就送標註後的圖，讓 PE 看得到圈選位置
            main_img = composite_img if (has_drawn and composite_img is not None) else bg_img
            if main_img is not None:
                imgs.append(main_img)
            imgs.extend(refs)
            if not imgs:
                return gr.update(), gr.update(), "⚠️ 改圖模式需要至少一張圖片才能改寫。"

        try:
            r = call_pe(task, p_text, imgs, sd)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", str(e))
            except Exception:
                msg = str(e)
            return gr.update(), gr.update(), f"❌ PE 服務錯誤：{msg}"
        except Exception as e:
            return gr.update(), gr.update(), (
                f"❌ 無法連線 PE 服務 ({PE_HOST})：{type(e).__name__}: {e}")

        new_prompt = r.get("positive_prompt", "")
        ok = r.get("parse_ok")
        ratio = (r.get("wh_ratio") or "").strip()
        follow = (r.get("ratio_follow") or "").strip()

        # 官方語意：t2i 用 wh_ratio 決定畫布；edit 若有 ratio_follow 則沿用該輸入圖比例
        ratio_update = gr.update()
        note = ""
        if task == "t2i" and ratio in PE_RATIO_TO_CHOICE:
            ratio_update = gr.update(value=PE_RATIO_TO_CHOICE[ratio])
            note = f"｜畫布 {ratio}"
        elif task == "edit":
            note = f"｜沿用 {follow} 的比例" if follow else (f"｜建議 {ratio}" if ratio else "")

        flag = "✅" if ok else "⚠️ 未解析出 JSON（已回填原始輸出）"
        return new_prompt, ratio_update, (
            f"{flag} PE 改寫完成（{task}，{r.get('elapsed')}s，"
            f"thinking {r.get('thinking_chars')} 字）{note}")

    btn_pe.click(
        fn=enhance_prompt,
        inputs=[mode, prompt, editor_input, ref_images_state, seed],
        outputs=[prompt, aspect_ratio, pe_status]
    )

    btn_pack.click(fn=pack_outputs, inputs=[pack_days],
                   outputs=[pack_file, pack_status])
    btn_clean.click(fn=cleanup_keep_recent, inputs=[keep_out, keep_in, confirm_del],
                    outputs=[clean_status])
    btn_purge.click(fn=purge_all, inputs=[purge_text], outputs=[purge_status])

    run_btn.click(
        fn=send_to_comfy,
        inputs=[
            mode, 
            edit_submode, 
            prompt, 
            neg_prompt, 
            editor_input, 
            ref_images_state, 
            aspect_ratio, 
            steps, 
            cfg, 
            seed,
            precision
        ],
        outputs=output_img
    )

# 注入針對不同區域的精準貼上偵測指令碼
paste_helper_script = """
<script>
window.lastHoverBox = 'main';

document.addEventListener('mouseover', (e) => {
    const refEl = document.getElementById('ref_paste_box');
    const mainEl = document.getElementById('main_canvas_box');
    if (refEl && refEl.contains(e.target)) {
        window.lastHoverBox = 'ref';
    } else if (mainEl && mainEl.contains(e.target)) {
        window.lastHoverBox = 'main';
    }
});

window.addEventListener('paste', (e) => {
    const active = document.activeElement;
    if (active && (active.tagName === 'TEXTAREA' || active.tagName === 'INPUT')) {
        return;
    }
    
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    let imageFile = null;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            imageFile = items[i].getAsFile();
            break;
        }
    }
    if (imageFile) {
        let dest = null;
        if (window.lastHoverBox === 'ref') {
            const refEl = document.getElementById('ref_paste_box');
            dest = refEl ? refEl.querySelector('input[type="file"]') : null;
        }
        if (!dest) {
            const mainEl = document.getElementById('main_canvas_box');
            dest = mainEl ? mainEl.querySelector('input[type="file"]') : null;
        }
        if (!dest) {
            const all = document.querySelectorAll('input[type="file"]');
            if (all.length > 0) dest = all[0];
        }
        
        if (dest) {
            const dt = new DataTransfer();
            dt.items.add(imageFile);
            dest.files = dt.files;
            dest.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
});
</script>
"""

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, head=paste_helper_script)
