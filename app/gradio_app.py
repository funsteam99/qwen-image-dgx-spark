import gradio as gr
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()  # iPhone HEIC/HEIF 照片支援
except ImportError:
    pass
import urllib.request
import urllib.error
import json
import io
import time
import os
import base64
import shutil
import threading
import tempfile
import zipfile
import numpy as np
import scipy.ndimage
from PIL import Image, ImageChops, ImageDraw, ImageFont

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
    # Turbo (2026-10-09)：同一個 7B 模型蒸餾成 8 步、CFG 1，文字編碼器與 VAE 與 2.1 共用。
    "turbo int8 (8 步加速版，省記憶體)": (
        "qwen_image_2.1_turbo_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors"),
    "turbo bf16 (8 步加速版)": (
        "qwen_image_2.1_turbo_bf16.safetensors", "qwen3vl_8b_int8_convrot.safetensors"),
}
# Turbo 的 8 步排程寫在權重裡，官方未評估其他排程；以 ManualSigmas 原樣送入。
TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"


def is_turbo(precision):
    return "turbo" in WEIGHT_PRESETS.get(precision, ("",))[0]
DEFAULT_PRECISION = "turbo int8 (8 步加速版，省記憶體)"
PE_HOST = "127.0.0.1:8200"   # 官方 Prompt Enhancer 常駐服務 (pe/pe_server.py)

# --- Prompt 改寫後端 ---------------------------------------------------------
# 本機已有的 vLLM 服務 (qwen3.8-27B，具視覺能力) 可直接擔任改寫器：
# 實測 t2i 28.2s / edit 17.6s，皆 parse_ok=True，且零額外記憶體與磁碟。
# 官方 PE 模型走 transformers batch-1，同樣任務需 144~160s，另需 19GB 常駐。
# 官方雖警告非微調模型未必穩定遵守輸出格式，故保留其為可選後端。
# 兩個本機 LLM 端點，2026-09-24 起皆已具備視覺能力（8002 原本無 mmproj 送圖回
# HTTP 500，已於服務端更新補上）：
#   8002  Qwen3.6-35B GGUF (llama.cpp)  文生圖改寫實測 8.2 秒，速度較快
#   8006  qwen3.8-27B-NVFP4 (vLLM)      文生圖約 22~31 秒、改圖約 17.6 秒
# 兩者速度/品質仍可能有差異，故不再自動路由，預設固定用 8002，由使用者視需要手動切換。
LLM_ENDPOINTS = {
    "8002": {"host": os.environ.get("LLM_HOST_TEXT", "127.0.0.1:8002"),
             "model": os.environ.get("LLM_MODEL_TEXT",
                                     "Qwen3.6-35B-A3B-abliterated.i1-Q4_K_M.gguf"),
             "vision": True, "label": "8002 Qwen3.6 (具視覺, 最快)"},
    "8006": {"host": os.environ.get("LLM_HOST_VL", "127.0.0.1:8006"),
             "model": os.environ.get("LLM_MODEL_VL", "qwen3.8"),
             "vision": True, "label": "8006 qwen3.8 (具視覺, 較慢)"},
}
PE_SYSTEM_PROMPTS = {
    "t2i": "/workspace/ComfyUI/pe/prompts/system_prompt_t2i.txt",
    "edit": "/workspace/ComfyUI/pe/prompts/system_prompt_edit.txt",
}
# 官方 PE 模型的呼叫路徑 (call_pe / pe_server.py) 保留於程式中，
# 但不列入選單：實測本機端點快 8~10 倍且零額外資源，PE 服務已停用。
PE_BACKENDS = {
    "8002 Qwen3.6 (具視覺, 最快)": "8002",
    "8006 qwen3.8 (具視覺, 較慢)": "8006",
}
DEFAULT_PE_BACKEND = "8002 Qwen3.6 (具視覺, 最快)"


def pick_endpoint(backend, task):
    """依後端選擇與任務決定端點。回傳 (key, config, 錯誤訊息)。"""
    need_vision = (task == "edit")
    cfg = LLM_ENDPOINTS[backend]
    if need_vision and not cfg["vision"]:
        return backend, cfg, f"⚠️ {cfg['label']} 不支援影像輸入，改圖模式請改選其他後端。"
    return backend, cfg, None

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


def _read_system_prompt(task):
    with open(PE_SYSTEM_PROMPTS[task], encoding="utf-8") as f:
        return f.read()


def call_llm(task, user_prompt, pil_images=None, endpoint=None, timeout=600):
    """用本機 vLLM 服務改寫，套用官方 system prompt 與官方的解析規則。

    enable_thinking 必須關閉：開啟時輸出常被 max_tokens 截斷在 thinking 階段，
    回傳空的 content（實測 259 秒且完全無輸出）。
    """
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + pil_to_b64(im)}}
               for im in (pil_images or [])]
    content.append({"type": "text", "text": user_prompt})
    endpoint = endpoint or LLM_ENDPOINTS["8006"]
    # 8002 (Qwen3.6 GGUF) 實測若只靠 system_prompt_*.txt 約束，
    # 輸出語言與格式會不穩定：時而中文、時而混雜，時而整包 JSON 原樣洩漏到 rewritten_prompt 之外。
    # 故在 system prompt 後方追加強制規則，覆寫模型自由發揮的空間。
    system_text = (
        _read_system_prompt(task)
        + "\n\nIMPORTANT OUTPUT RULES (override anything above if conflicting):\n"
        "1. Output EXACTLY one JSON object and nothing else — no markdown code "
        "fences, no explanation before or after, no extra JSON objects.\n"
        "2. The value of \"rewritten_prompt\" MUST be written entirely in English, "
        "regardless of what language the user's input prompt is in.\n"
    )
    body = {
        "model": endpoint["model"],
        "messages": [{"role": "system", "content": system_text},
                     {"role": "user", "content": content}],
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        # 官方 pe_core profile：t2i 用 1.5、edit 用 0.0。
        # 漏掉此值時 8002 的輸出長度在 724~1479 字元間大幅浮動；補上後穩定於 1600 上下。
        "presence_penalty": 1.5 if task == "t2i" else 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://{endpoint['host']}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data["choices"][0]["message"].get("content") or "")
    parsed = _parse_pe_answer(text, task)
    parsed["elapsed"] = round(time.time() - t0, 1)
    parsed["thinking_chars"] = 0
    return parsed


def _parse_pe_answer(answer, task):
    """比照官方 pe_core.parse_answer：由後往前找最後一個合法 JSON 物件。"""
    import re
    answer = (answer or "").strip()
    # 有時模型會用 ```json ... ``` 包住輸出，先拆掉圍籬避免干擾大括號配對
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.MULTILINE).strip()
    spans = []
    depth = start = 0
    for i, ch in enumerate(answer):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                spans.append(answer[start:i + 1])
    for cand in reversed(spans):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if isinstance(rewritten, str) and rewritten.strip():
            return {"positive_prompt": rewritten.strip(),
                    "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
                    "ratio_follow": (str(obj.get("ratio_follow") or "").strip()
                                     if task == "edit" else ""),
                    "parse_ok": True}
    return {"positive_prompt": answer, "wh_ratio": "", "ratio_follow": "",
            "parse_ok": False}


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
    precision=DEFAULT_PRECISION,
    original_prompt="",
    my_jobs=None
):
    # 「擴寫後」欄位若是空的（例如使用者只打了原始提示詞、還沒按「Prompt 改寫」就直接生成），
    # 就退回用「原始提示詞」欄位的內容送去生成，避免送出空提示詞。
    # 記下 was_rewritten，讓「回看」能區分「真的擴寫過」和「只是退回原始提示詞」。
    prompt = (prompt or "").strip()
    was_rewritten = bool(prompt)
    prompt = prompt or (original_prompt or "").strip()
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
    if is_turbo(precision):
        # Turbo：固定官方 8 步 sigmas 與 CFG 1，忽略步數/CFG 滑桿。
        workflow["20"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": TURBO_SIGMAS}}
        workflow["21"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        workflow["6"] = {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["9", 0],
                "add_noise": True,
                "noise_seed": workflow["6"]["inputs"]["seed"],
                "cfg": 1.0,
                "positive": ["4", 0],
                "negative": ["4", 1],
                "sampler": ["21", 0],
                "sigmas": ["20", 0],
                "latent_image": latent_source,
            }
        }
        steps = 8
    workflow["7"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["6", 0],
            "vae": ["3", 0]
        }
    }
    # 檔名帶時間戳：清理輸出後 ComfyUI 的流水號會從 00001 重來，
    # 固定前綴會讓新檔覆用舊檔名，使「回看」的舊紀錄指向內容已不同的圖（靜默錯誤）。
    workflow["8"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": time.strftime("QwenImage21_%Y%m%d_%H%M%S"),
            "images": ["7", 0]
        }
    }

    req_data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=req_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    prompt_id = res.get("prompt_id")
    if not prompt_id:
        raise gr.Error(f"ComfyUI 拒絕了這個工作：{json.dumps(res, ensure_ascii=False)[:300]}")

    # 送出不等結果：結果由背景執行緒收，紀錄照寫，關掉分頁也不會漏。
    meta = {
        "mode": mode,
        "seed": workflow["6"]["inputs"].get("seed", workflow["6"]["inputs"].get("noise_seed")),
        "steps": int(steps),
        "precision": unet_name,
        "was_rewritten": was_rewritten,
    }
    with JOBS_LOCK:
        JOBS[prompt_id] = {"status": "queued", "submitted": time.time(),
                           "prompt": prompt, "file": None, "error": None}
    threading.Thread(target=_track_job, args=(prompt_id, original_prompt, prompt, meta),
                     daemon=True).start()

    my_jobs = list(my_jobs or []) + [prompt_id]
    return my_jobs, f"✅ 已排入佇列（{prompt_id[:8]}）。可繼續送出下一張，進度看下方「佇列狀態」。"


# =============================================================================
# 背景追蹤：送出後不等結果，由執行緒輪詢 ComfyUI /history 收結果、寫紀錄。
# 容器重啟時執行緒會消失，尚未完成的工作只剩輸出目錄裡的圖檔，不會進紀錄。
# =============================================================================
JOBS = {}                 # prompt_id -> {status, submitted, prompt, file, error}
JOBS_LOCK = threading.Lock()
JOBS_KEEP = 50


def _comfy_json(path, data=None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(f"http://{COMFY_HOST}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def _set_job(prompt_id, **kw):
    with JOBS_LOCK:
        if prompt_id in JOBS:
            JOBS[prompt_id].update(kw)
        # 只留最近的，避免長時間運行後無限成長
        if len(JOBS) > JOBS_KEEP:
            for k in sorted(JOBS, key=lambda k: JOBS[k]["submitted"])[:len(JOBS) - JOBS_KEEP]:
                if JOBS[k]["status"] in ("done", "failed", "cancelled"):
                    JOBS.pop(k, None)


def _track_job(prompt_id, original_prompt, prompt, meta):
    missing = 0
    while True:
        time.sleep(2)
        try:
            entry = _comfy_json(f"/history/{prompt_id}").get(prompt_id)
            if entry is None:
                # 不在紀錄也不在佇列 → 已被從佇列刪除（取消）
                q = _comfy_json("/queue")
                running = {it[1] for it in q.get("queue_running", [])}
                pending = {it[1] for it in q.get("queue_pending", [])}
                if prompt_id in running or prompt_id in pending:
                    missing = 0
                    _set_job(prompt_id, status="running" if prompt_id in running else "queued")
                    continue
                missing += 1
                if missing >= 3:
                    _set_job(prompt_id, status="cancelled")
                    return
                continue
            outputs = entry.get("outputs", {})
            if "8" in outputs and outputs["8"].get("images"):
                info = outputs["8"]["images"][0]
                rel = os.path.join(info.get("subfolder") or "", info["filename"])
                size = ""
                try:
                    with Image.open(os.path.join(COMFY_OUTPUT_DIR, rel)) as im:
                        size = f"{im.width}x{im.height}"
                except Exception:
                    pass
                log_generation(original_prompt, prompt, rel, {**meta, "size": size})
                _set_job(prompt_id, status="done", file=rel)
                return
            status = entry.get("status", {})
            if status.get("status_str") and status.get("status_str") != "success":
                msgs = json.dumps(status.get("messages", [])[-2:], ensure_ascii=False)
                interrupted = "execution_interrupted" in msgs
                _set_job(prompt_id, status="cancelled" if interrupted else "failed",
                         error=None if interrupted else msgs[:300])
                return
        except Exception as e:
            print(f"[track {prompt_id[:8]}] {type(e).__name__}: {e}")


def cancel_job(prompt_id):
    """排隊中的從佇列刪除；執行中的中斷。整個 ComfyUI 佇列都可操作。"""
    prompt_id = (prompt_id or "").strip()
    if not prompt_id:
        return
    try:
        q = _comfy_json("/queue")
        if any(it[1] == prompt_id for it in q.get("queue_running", [])):
            _comfy_json("/interrupt", {"prompt_id": prompt_id})
            gr.Info("已送出中斷。")
        elif any(it[1] == prompt_id for it in q.get("queue_pending", [])):
            _comfy_json("/queue", {"delete": [prompt_id]})
            gr.Info("已從佇列移除。")
        else:
            gr.Warning("這個工作已經不在佇列中。")
    except Exception as e:
        raise gr.Error(f"取消失敗：{type(e).__name__}: {e}")


def _short(text, n=60):
    return text[:n] + ("…" if len(text) > n else "")


def _job_label(item):
    """從 ComfyUI 佇列項目的工作流取出提示詞前段，當作顯示名稱。"""
    try:
        for node in item[2].values():
            p = node.get("inputs", {}).get("prompt")
            if isinstance(p, str):
                return _short(p)
    except Exception:
        pass
    return "（非文字工作流）"


def build_queue_view(my_jobs):
    """佇列狀態：整個 ComfyUI 佇列 + 7860 最近結束的工作。回傳 (html, 本頁最新完成的 job id)。

    由前端 JS 輪詢 /queue_view 呼叫，不走 Gradio 事件：Gradio 每個事件的狀態變化都會向
    「曾參與事件的元件」重新取值，主圖 ImageEditor 每取一次值就把整張圖重新上傳一次。
    """
    try:
        q = _comfy_json("/queue")
    except Exception as e:
        return f'<p style="color:#b91c1c">無法讀取 ComfyUI 佇列：{_esc(str(e))}</p>', None
    running = q.get("queue_running", [])
    pending = sorted(q.get("queue_pending", []), key=lambda it: it[0])
    mine = set(my_jobs or [])

    def row(icon, state, item_id, label, action):
        who = "🙋 我" if item_id in mine else ""
        btn = (f'<button class="queue-cancel" data-id="{_esc(item_id)}">{action}</button>'
               if action else "")
        return (f"<tr><td style='white-space:nowrap'>{icon} {state}</td><td>{_esc(label)}</td>"
                f"<td style='white-space:nowrap'>{who}</td><td>{btn}</td></tr>")

    rows = [row("⏳", "執行中", it[1], _job_label(it), "中斷") for it in running]
    rows += [row("🕒", f"排隊第 {n}", it[1], _job_label(it), "取消")
             for n, it in enumerate(pending, 1)]

    with JOBS_LOCK:
        finished = sorted(((k, dict(v)) for k, v in JOBS.items()
                           if v["status"] in ("done", "failed", "cancelled")),
                          key=lambda kv: kv[1]["submitted"], reverse=True)[:5]
    icons = {"done": ("✅", "完成"), "failed": ("❌", "失敗"), "cancelled": ("🚫", "已取消")}
    for k, v in finished:
        ic, st = icons[v["status"]]
        label = _short(v["prompt"]) + (f"｜{v['error'][:120]}" if v["error"] else "")
        rows.append(row(ic, st, k, label, None))

    html = f"<p>執行中 {len(running)}、排隊 {len(pending)}。已結束的只列本頁面（7860）送出的最近 5 張。</p>"
    if rows:
        html += ('<table class="review-table"><thead><tr><th style="width:110px">狀態</th>'
                 "<th>提示詞</th><th style='width:50px'></th><th style='width:60px'></th></tr></thead>"
                 "<tbody>" + "".join(rows) + "</tbody></table>")

    # 輸出區：這個頁面送出的工作裡，最新完成的那張（由前端決定是否載入）
    with JOBS_LOCK:
        newest = next((k for k in reversed(my_jobs or [])
                       if k in JOBS and JOBS[k]["status"] == "done"), None)
    return html, newest


def load_job_result(job_id):
    """輸出區載入指定工作的結果圖；只接受 JOBS 內已完成的 id，不接受前端傳路徑。"""
    job_id = (job_id or "").strip()
    with JOBS_LOCK:
        job = dict(JOBS[job_id]) if job_id in JOBS else None
    if not job or job["status"] != "done":
        return gr.update()
    try:
        return Image.open(os.path.join(COMFY_OUTPUT_DIR, job["file"]))
    except Exception:
        return gr.update()


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
# 回看：生成歷史
# ComfyUI 的 /history 只保留「實際送出」的提示詞（即擴寫後的），
# 原始短句只存在於前端，故在此自行記錄成 JSONL。
# =============================================================================
HISTORY_LOG = os.environ.get("HISTORY_LOG", "/workspace/generation_history.jsonl")
HISTORY_MAX_ROWS = 30
THUMB_WIDTH = 220


def log_generation(original_prompt, final_prompt, filename, meta):
    """附加一筆生成紀錄。失敗不得影響出圖，故整段吞例外。"""
    try:
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original": (original_prompt or "").strip(),
            "final": (final_prompt or "").strip(),
            "file": filename,
            **meta,
        }
        with open(HISTORY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _thumb_data_uri(filename):
    """讀取輸出圖並縮成縮圖的 base64 data URI。"""
    path = os.path.join(COMFY_OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        return None
    im = Image.open(path)
    im.thumbnail((THUMB_WIDTH, THUMB_WIDTH), Image.LANCZOS)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _esc(t):
    return (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_history_html(limit=HISTORY_MAX_ROWS):
    """組出回看表格。縮圖內嵌為 data URI，點擊則以 Gradio 檔案端點開啟原圖。"""
    if not os.path.isfile(HISTORY_LOG):
        return "<p>尚無紀錄。生成一張圖後按「🔄 重新整理」即可看到。</p>"
    rows = []
    with open(HISTORY_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = rows[::-1][:int(limit or HISTORY_MAX_ROWS)]
    if not rows:
        return "<p>尚無紀錄。</p>"

    html = ['<div style="overflow-x:auto"><table class="review-table">',
            '<thead><tr><th style="width:130px">時間</th>'
            '<th style="width:22%">原始提示</th><th>擴寫提示</th>'
            f'<th style="width:{THUMB_WIDTH + 20}px">圖片</th></tr></thead><tbody>']
    for r in rows:
        fn = r.get("file") or ""
        uri = _thumb_data_uri(fn)
        full = f"/gradio_api/file={COMFY_OUTPUT_DIR}/{fn}"
        if uri:
            img = (f'<a href="{full}" target="_blank" title="點擊看原圖">'
                   f'<img src="{uri}" style="max-width:100%;border-radius:6px"></a>')
        else:
            img = '<span style="opacity:.5">（檔案已清理）</span>'
        if uri:
            # 按鈕由 head 腳本以事件委派接手，轉交給隱藏的橋接元件
            img += (f'<div style="margin-top:6px;display:flex;gap:6px">'
                    f'<button class="hist-send" data-dest="main" data-file="{_esc(fn)}">送到主圖</button>'
                    f'<button class="hist-send" data-dest="ref" data-file="{_esc(fn)}">加到參考圖</button></div>')
        orig = _esc(r.get("original")) or '<span style="opacity:.5">（未改寫）</span>'
        meta = f'{r.get("mode","")} · {r.get("size","")} · seed {r.get("seed","")}'
        if r.get("was_rewritten"):
            final = _esc(r.get("final"))
        else:
            final = '<span style="opacity:.5">（未改寫）</span>'
        html.append(
            "<tr>"
            f'<td style="white-space:nowrap;vertical-align:top">{_esc(r.get("ts"))}'
            f'<div style="opacity:.55;font-size:.82em;margin-top:4px">{_esc(meta)}</div></td>'
            f'<td style="vertical-align:top">{orig}</td>'
            f'<td style="vertical-align:top">{final}</td>'
            f'<td style="vertical-align:top">{img}</td>'
            "</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


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


# =============================================================================
# 歷史圖片來源：把 ComfyUI 輸出目錄裡還在的圖，送回主圖或參考圖
# 掃整個輸出目錄（含 8188 直接生成的），不只 7860 的 jsonl 紀錄。
# =============================================================================
RUN_BTN_LABEL = "🚀 開始執行生成 / 替換 (Execute)"
PICKER_PAGE = 24          # 浮動視窗每批載入幾張，捲到底再載下一批
PICKER_THUMB = 256
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_picker_thumb_cache = {}   # (path, mtime) -> 縮圖；2K 原圖直接丟進 Gallery 太重


def _scan_output_images():
    """遞迴列出輸出目錄中的圖片，新到舊。只取路徑與時間，不開圖檔。"""
    found = []
    if os.path.isdir(COMFY_OUTPUT_DIR):
        for root, _dirs, names in os.walk(COMFY_OUTPUT_DIR):
            for name in names:
                if name in PROTECTED_NAMES or not name.lower().endswith(IMAGE_EXTS):
                    continue
                fp = os.path.join(root, name)
                try:
                    found.append((os.path.getmtime(fp), fp))
                except OSError:
                    continue
    found.sort(reverse=True)
    return [fp for _m, fp in found]


def _picker_thumb(fp):
    key = (fp, os.path.getmtime(fp))
    im = _picker_thumb_cache.get(key)
    if im is None:
        im = Image.open(fp)
        im.thumbnail((PICKER_THUMB, PICKER_THUMB), Image.LANCZOS)
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        _picker_thumb_cache[key] = im
    return im


PICK_REF_COLOR = (37, 99, 235)      # 藍：參考圖，依點選順序編號
PICK_BORDER = 16
PICK_TARGET_MAIN = "main"
PICK_TARGET_REF = "ref"


def _badge_font():
    try:
        return ImageFont.load_default(size=40)
    except TypeError:                # Pillow < 10.1 沒有 size 參數
        return ImageFont.load_default()


def _mark_thumb(im, color, badge=None):
    """選取標示直接畫進縮圖：外框 + 左上角序號。"""
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    for k in range(PICK_BORDER):
        d.rectangle([k, k, w - 1 - k, h - 1 - k], outline=color)
    if badge is not None:
        d.rectangle([0, 0, 64, 56], fill=color)
        d.text((32, 28), str(badge), fill="white", font=_badge_font(), anchor="mm")
    return im


def _picker_items(paths, sel, shown):
    """只產生前 shown 張的縮圖；縮圖在第一次載到時才做並快取。"""
    sel = sel or []
    items = []
    for i, fp in enumerate(paths[:shown]):
        try:
            thumb = _picker_thumb(fp)
            t = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(fp)))
            if i in sel:
                n = sel.index(i) + 1
                items.append((_mark_thumb(thumb, PICK_REF_COLOR, n), f"🟦 {n} · {t}"))
            else:
                items.append((thumb, t))
        except Exception:
            items.append((Image.new("RGB", (PICKER_THUMB, PICKER_THUMB), (200, 200, 200)), "（已清理）"))
    return items


def _picker_status(target, sel, paths, n_refs, shown):
    if not paths:
        return "輸出目錄沒有圖片。"
    loaded = f"已載入 {min(shown, len(paths))} / {len(paths)} 張（捲到底自動載入更多）"
    if target == PICK_TARGET_MAIN:
        return f"{loaded}。**點一張**就直接設為主圖並關閉視窗。"
    room = max(0, 10 - n_refs)
    return (f"{loaded}。點縮圖依序選取（再點一次取消），已選 **{len(sel or [])}** 張；"
            f"目前參考圖 {n_refs} 張，還能加 {room} 張。選好按「✅ 加入參考圖」。")


def open_picker(target, refs):
    """打開浮動視窗：掃描輸出目錄（只列路徑），縮圖只做第一批。"""
    paths = _scan_output_images()
    shown = PICKER_PAGE
    title = ("### 🕘 選擇過往圖片 → 🖼️ 主圖" if target == PICK_TARGET_MAIN
             else "### 🕘 選擇過往圖片 → 📎 參考圖（可多選）")
    return (gr.update(visible=True), title, _picker_items(paths, [], shown), paths, [], target,
            _picker_status(target, [], paths, len(refs or []), shown),
            gr.update(visible=(target == PICK_TARGET_REF)), shown)


def picker_load_more(target, sel, paths, refs, shown):
    if shown >= len(paths or []):
        return gr.update(), gr.update(), shown
    shown += PICKER_PAGE
    return (_picker_items(paths, sel, shown),
            _picker_status(target, sel, paths, len(refs or []), shown), shown)


def _resolve_output(fp):
    """只接受輸出目錄內、仍存在的檔案（橋接元件的值來自前端，不可信）。"""
    if not fp:
        return None
    if not os.path.isabs(fp):
        fp = os.path.join(COMFY_OUTPUT_DIR, fp)
    real = os.path.realpath(fp)
    base = os.path.realpath(COMFY_OUTPUT_DIR)
    if os.path.commonpath([real, base]) != base or not os.path.isfile(real):
        return None
    return real


def _refs_out(refs):
    return [(img, f"<image{i+1}>") for i, img in enumerate(refs)]


def send_paths_to_main(fps, cur_mode):
    """第一張送到主圖並切到「以圖改圖」。原有畫筆標註一併清掉。"""
    fps = [p for p in (_resolve_output(f) for f in fps) if p]
    if not fps:
        gr.Warning("沒有可用的圖片（未選取或檔案已被清理）。")
        return (gr.update(),) * 5
    img = Image.open(fps[0]).convert("RGBA")
    if len(fps) > 1:
        gr.Info("主圖只取選取的第一張。")
    return ({"background": img, "layers": [], "composite": img},
            MODE_EDIT, gr.Tabs(selected=MODE_EDIT),
            gr.update(visible=True), gr.update(visible=True))


def send_paths_to_refs(fps, refs, cur_mode):
    """接在現有參考圖之後，最多 10 張。純文字模式下會切到「多圖參考融合」。"""
    refs = list(refs or [])
    fps = [p for p in (_resolve_output(f) for f in fps) if p]
    if not fps:
        gr.Warning("沒有可用的圖片（未選取或檔案已被清理）。")
        return (gr.update(),) * 6
    room = 10 - len(refs)
    if room <= 0:
        gr.Warning("最多支援 10 張參考圖！已達到上限。")
        return (gr.update(),) * 6
    if len(fps) > room:
        gr.Warning(f"參考圖上限 10 張，只加入前 {room} 張。")
    for fp in fps[:room]:
        refs.append(Image.open(fp).convert("RGB"))
    new_mode = MODE_MULTI if cur_mode == MODE_T2I else cur_mode
    return (_refs_out(refs), refs, new_mode, gr.Tabs(selected=new_mode),
            gr.update(visible=(new_mode == MODE_EDIT)), gr.update(visible=True))


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
.hist-bridge { display: none !important; }
#picker_gallery img { object-fit: contain !important; }
#history_modal {
    position: fixed !important; top: 4vh; left: 50%; transform: translateX(-50%);
    width: min(1200px, 94vw); max-height: 92vh; overflow: auto; z-index: 1000;
    background: var(--background-fill-primary); border-radius: 12px; padding: 16px;
    box-shadow: 0 0 0 100vmax rgba(0, 0, 0, .55), 0 10px 40px rgba(0, 0, 0, .4);
}
'''

with gr.Blocks(title="Qwen-Image-2.1 官方標準工作站 (DGX Spark)", css=custom_css) as demo:
    ref_images_state = gr.State([])
    # 保存改寫前的原始提示詞，供「回看」對照用
    
    # --- 過往圖片浮動視窗：主圖 / 參考圖各自的按鈕打開，打開時才掃描輸出目錄 ---
    picker_paths = gr.State([])
    picker_sel = gr.State([])
    picker_shown = gr.State(0)
    picker_target = gr.State(PICK_TARGET_MAIN)
    with gr.Column(visible=False, elem_id="history_modal") as picker_modal:
        with gr.Row():
            picker_title = gr.Markdown("### 🕘 選擇過往圖片", elem_id="history_modal_title")
            btn_picker_close = gr.Button("✕ 關閉", size="sm", scale=0, min_width=90)
        picker_status = gr.Markdown("")
        picker_gallery = gr.Gallery(
            show_label=False, columns=6, height="65vh", allow_preview=False,
            elem_id="picker_gallery", interactive=False
        )
        with gr.Row():
            btn_picker_confirm = gr.Button("✅ 加入參考圖", variant="primary", visible=False)
            # 捲到底時由 head 腳本自動點擊；手動點也可以
            btn_picker_more = gr.Button("⬇️ 載入更多", elem_id="picker_more_btn")
            btn_picker_cancel = gr.Button("取消")

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
            # 記住最後選的功能分頁（存在瀏覽器 localStorage，重新整理後還原）
            mode_store = gr.BrowserState(MODE_EDIT, storage_key="qwen_last_mode")
            with gr.Tabs() as mode_tabs:
                with gr.TabItem("🎯 以圖改圖", id=MODE_EDIT):
                    gr.Markdown("上傳主圖後下指令。可用筆刷圈選要修改的位置，並加掛最多 10 張參考圖。")
                with gr.TabItem("🖼️ 多圖參考融合", id=MODE_MULTI):
                    gr.Markdown("貼上多張參考圖，用 `<image1>`、`<image2>` … 在提示詞中指名合成。")
                with gr.TabItem("✨ 純文字生圖", id=MODE_T2I):
                    gr.Markdown("不需圖片，直接輸入提示詞。尺寸由右側「生成參數」決定。")
            
            # 回看表格內按鈕的橋接元件：以 CSS 隱藏（visible=False 不會出現在 DOM）
            with gr.Row(elem_classes=["hist-bridge"]):
                hist_bridge_file = gr.Textbox(elem_id="hist_bridge_file", show_label=False)
                hist_bridge_main = gr.Button("main", elem_id="hist_bridge_main")
                hist_bridge_ref = gr.Button("ref", elem_id="hist_bridge_ref")

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
                btn_pick_main = gr.Button("🕘 從過往圖片選擇主圖", size="sm")
                
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
                        btn_pick_refs = gr.Button("🕘 從過往圖片加入參考圖", size="sm")
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

            orig_prompt_display = gr.Textbox(
                label="📝 原始提示詞（手動輸入，每次按「Prompt 改寫」都以這裡的內容為準）",
                placeholder="在這裡輸入你的指令，例如：一隻橘貓坐在窗台上看著窗外下雨，水彩畫風格",
                lines=2
            )

            prompt = gr.Textbox(
                label="✨ 擴寫後提示詞 (Prompt - 會送去生成；建議依官方規範使用 <image1>, <image2> 錨定)",
                placeholder="按上方輸入原始提示詞後，點「Prompt 改寫」產生的結果會顯示於此；也可直接在此手動輸入",
                lines=3
            )

            with gr.Row():
                btn_pe = gr.Button("✨ Prompt 改寫", variant="secondary", size="lg")
                pe_backend = gr.Dropdown(
                    choices=list(PE_BACKENDS.keys()),
                    value=DEFAULT_PE_BACKEND,
                    label="改寫引擎",
                    scale=2
                )
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
            
            run_btn = gr.Button(RUN_BTN_LABEL, variant="primary", size="lg", elem_id="run_btn")

        with gr.Column(scale=4):
            output_img = gr.Image(
                label="🖼️ 輸出結果 (新圖像)",
                type="pil",
                image_mode="RGBA",
                elem_classes=["output-checkerboard"]
            )
            submit_status = gr.Markdown("")
            gr.Markdown("### 📋 佇列狀態（整個 ComfyUI 佇列，每 3 秒更新）")
            # 內容由前端 JS 輪詢 /queue_view 填入（見 paste_helper_script），不用 gr.Timer
            queue_html = gr.HTML('<div id="queue_view"><p>載入中…</p></div>')
            my_jobs = gr.State([])
            with gr.Row(elem_classes=["hist-bridge"]):
                cancel_bridge_id = gr.Textbox(elem_id="cancel_bridge_id", show_label=False)
                cancel_bridge_btn = gr.Button("cancel", elem_id="cancel_bridge_btn")
                # 本頁送出的 job id（逗號分隔），給前端輪詢用
                my_jobs_box = gr.Textbox(elem_id="my_jobs_box", show_label=False)
                # 前端發現新完成的工作時，寫入 id 並點按鈕把結果載入輸出區
                result_bridge_id = gr.Textbox(elem_id="result_bridge_id", show_label=False)
                result_bridge_btn = gr.Button("result", elem_id="result_bridge_btn")

    # 生成參數置於頁面最下方：平時不佔版面，需要時捲到底調整。
    # ⚠️ 必須保持展開：Gradio 6 的折疊 Accordion 不渲染子元件，
    # 提交時會送出預設值而非畫面上的值（實測 seed 顯示官方值、送出卻是 -1）。
    # 這些控制項是 run_btn 的 inputs，故不可折疊。
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

        def on_precision_change(p):
            # Turbo 的步數與 CFG 由權重決定，滑桿顯示實際值並鎖定，避免誤以為可調。
            if is_turbo(p):
                return (gr.update(minimum=1, value=8, interactive=False,
                                  label="推論步數 (Turbo 固定 8 步)"),
                        gr.update(value=1.0, interactive=False))
            return (gr.update(minimum=15, value=40, interactive=True,
                              label="推論步數 (官方 README: num_inference_steps=40)"),
                    gr.update(interactive=True))
        precision.change(on_precision_change, inputs=precision, outputs=[steps, cfg])
        demo.load(on_precision_change, inputs=precision, outputs=[steps, cfg])

    with gr.Accordion("🕘 回看 (生成紀錄)", open=False):
        with gr.Row():
            hist_limit = gr.Number(value=10, precision=0, label="顯示最近幾筆", scale=1)
            btn_hist = gr.Button("🔄 重新整理", scale=1)
        hist_html = gr.HTML("<p>按「🔄 重新整理」載入紀錄。</p>")

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

    # 不論是點分頁、載入範例或從過往圖片切換，模式一變就記下來
    mode.change(fn=lambda m: m, inputs=[mode], outputs=[mode_store], queue=False)

    def restore_mode(saved):
        m = saved if saved in (MODE_EDIT, MODE_MULTI, MODE_T2I) else MODE_EDIT
        vis_canvas, vis_refs = on_mode_change(m)
        return m, gr.Tabs(selected=m), vis_canvas, vis_refs

    demo.load(fn=restore_mode, inputs=[mode_store],
              outputs=[mode, mode_tabs, edit_canvas_container, ref_images_container],
              queue=False)

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
        ).then(on_precision_change, inputs=precision, outputs=[steps, cfg], queue=False)
        # 範例寫死 40 步；選 Turbo 時再依精度覆寫回 8 步並鎖定。

    def enhance_prompt(m, orig_text, editor_data, refs, sd, backend_label):
        """官方 Prompt Enhancer：拿「原始提示詞」欄位當下的內容去改寫，
        每次按下都以該欄位為準，覆蓋上一次的擴寫結果；原始欄位本身永不被程式改動。
        """
        p_text = orig_text
        if not (p_text or "").strip():
            return gr.update(), gr.update(), "⚠️ 請先在「原始提示詞」欄位輸入內容再改寫。"

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

        backend = PE_BACKENDS.get(backend_label, "8002")
        endpoint = None
        if backend != "pe":
            _key, endpoint, warn = pick_endpoint(backend, task)
            if warn:
                return gr.update(), gr.update(), warn
        try:
            if backend == "pe":
                r = call_pe(task, p_text, imgs, sd)
            else:
                r = call_llm(task, p_text, imgs, endpoint=endpoint)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", str(e))
            except Exception:
                msg = str(e)
            return gr.update(), gr.update(), f"❌ 改寫服務錯誤（{backend}）：{msg}"
        except Exception as e:
            host = PE_HOST if backend == "pe" else endpoint["host"]
            return gr.update(), gr.update(), (
                f"❌ 無法連線改寫服務 ({host})：{type(e).__name__}: {e}")

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

        engine = "官方 PE" if backend == "pe" else endpoint["label"]
        if not ok:
            # 解析失敗時原始輸出可能混雜語言或殘留 JSON 片段，不可污染提示詞欄位，
            # 「擴寫後」欄位保持原樣不變，可疑輸出留在狀態列供排查。
            preview = new_prompt.replace("\n", " ")[:200]
            flag = "⚠️ 未解析出 JSON，「擴寫後」欄位維持不變"
            return gr.update(), ratio_update, (
                f"{flag}｜{engine}（{task}，{r.get('elapsed')}s）{note}｜原始輸出：{preview}")

        # 每次擴寫成功都覆蓋「擴寫後」欄位（不動「原始提示詞」欄位）
        return new_prompt, ratio_update, (
            f"✅ {engine} 改寫完成（{task}，{r.get('elapsed')}s）{note}")

    btn_pe.click(
        fn=enhance_prompt,
        inputs=[mode, orig_prompt_display, editor_input, ref_images_state, seed, pe_backend],
        outputs=[prompt, aspect_ratio, pe_status]
    )

    # 過往圖片浮動視窗
    _main_outputs = [editor_input, mode, mode_tabs, edit_canvas_container, ref_images_container]
    _ref_outputs = [ref_gallery, ref_images_state, mode, mode_tabs,
                    edit_canvas_container, ref_images_container]
    _open_outputs = [picker_modal, picker_title, picker_gallery, picker_paths, picker_sel,
                     picker_target, picker_status, btn_picker_confirm, picker_shown]

    btn_pick_main.click(fn=lambda refs: open_picker(PICK_TARGET_MAIN, refs),
                        inputs=[ref_images_state], outputs=_open_outputs)
    btn_pick_refs.click(fn=lambda refs: open_picker(PICK_TARGET_REF, refs),
                        inputs=[ref_images_state], outputs=_open_outputs)
    for _b in (btn_picker_close, btn_picker_cancel):
        _b.click(fn=lambda: gr.update(visible=False), outputs=[picker_modal], queue=False)

    btn_picker_more.click(
        fn=picker_load_more,
        inputs=[picker_target, picker_sel, picker_paths, ref_images_state, picker_shown],
        outputs=[picker_gallery, picker_status, picker_shown], trigger_mode="once")

    def on_pick(target, sel, paths, refs, m, shown, evt: gr.SelectData):
        """主圖：點一張直接套用並關窗。參考圖：切換選取並重畫編號。"""
        keep = (gr.update(),) * 5
        if target == PICK_TARGET_MAIN:
            if evt.index >= len(paths or []):
                return (sel, gr.update(), gr.update(), gr.update()) + keep
            res = send_paths_to_main([paths[evt.index]], m)
            return (sel, gr.update(), gr.update(), gr.update(visible=False)) + tuple(res)
        sel = list(sel or [])
        if evt.index in sel:
            sel.remove(evt.index)
        elif len(sel) + len(refs or []) >= 10:
            gr.Warning("參考圖上限 10 張，不能再選了。")
        else:
            sel.append(evt.index)
        return (sel, _picker_status(target, sel, paths, len(refs or []), shown),
                _picker_items(paths, sel, shown), gr.update()) + keep

    picker_gallery.select(
        fn=on_pick,
        inputs=[picker_target, picker_sel, picker_paths, ref_images_state, mode, picker_shown],
        outputs=[picker_sel, picker_status, picker_gallery, picker_modal] + _main_outputs,
        queue=False)

    def confirm_refs(sel, paths, refs, m):
        picked = [paths[i] for i in (sel or []) if i < len(paths or [])]
        if not picked:
            gr.Warning("尚未選取任何圖片。")
            return (gr.update(),) + (gr.update(),) * 6
        return (gr.update(visible=False),) + tuple(send_paths_to_refs(picked, refs, m))

    btn_picker_confirm.click(
        fn=confirm_refs,
        inputs=[picker_sel, picker_paths, ref_images_state, mode],
        outputs=[picker_modal] + _ref_outputs, queue=False)

    hist_bridge_main.click(
        fn=lambda f, m: send_paths_to_main([f], m),
        inputs=[hist_bridge_file, mode], outputs=_main_outputs, queue=False)
    hist_bridge_ref.click(
        fn=lambda f, refs, m: send_paths_to_refs([f], refs, m),
        inputs=[hist_bridge_file, ref_images_state, mode], outputs=_ref_outputs, queue=False)

    btn_hist.click(fn=build_history_html, inputs=[hist_limit],
                   outputs=[hist_html], queue=False)

    btn_pack.click(fn=pack_outputs, inputs=[pack_days],
                   outputs=[pack_file, pack_status])
    btn_clean.click(fn=cleanup_keep_recent, inputs=[keep_out, keep_in, confirm_del],
                    outputs=[clean_status])
    btn_purge.click(fn=purge_all, inputs=[purge_text], outputs=[purge_status])

    # 送出不等結果：按下即排入 ComfyUI 佇列並返回，可連續送出；結果由佇列狀態區接手。
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
            precision,
            orig_prompt_display,
            my_jobs
        ],
        outputs=[my_jobs, submit_status]
    ).then(fn=lambda j: ",".join(j or []), inputs=[my_jobs], outputs=[my_jobs_box], queue=False)

    cancel_bridge_btn.click(fn=cancel_job, inputs=[cancel_bridge_id], queue=False)
    result_bridge_btn.click(fn=load_job_result, inputs=[result_bridge_id], outputs=[output_img],
                            queue=False, show_progress="hidden")

# 注入針對不同區域的精準貼上偵測指令碼
paste_helper_script = """
<script>
// 上傳去重：Gradio 每個事件的狀態變化都會向主圖 ImageEditor 重新取值，而它每取一次就把
// 背景/合成/圖層整組重新上傳（iPhone 上一按生成就排了約 40 個 1.2MB 上傳）。
// 同一個檔案（SHA-1 + 大小 + 檔名相同）10 分鐘內只真的上傳一次，之後直接回傳上次的伺服器路徑；
// 同時進行中的相同上傳共用同一個請求。crypto.subtle 只在 HTTPS 可用，HTTP 下不去重。
(() => {
    if (!window.crypto || !crypto.subtle) return;
    const TTL = 10 * 60 * 1000, MAX = 30;
    const cache = new Map();  // key -> { t, text: Promise<string|null> }
    const origFetch = window.fetch;
    window.fetch = async function (input, init) {
        const url = typeof input === 'string' ? input : (input && input.url) || '';
        const body = init && init.body;
        if (!url.includes('/gradio_api/upload') || !(body instanceof FormData)) {
            return origFetch.apply(this, arguments);
        }
        const files = body.getAll('files');
        if (files.length !== 1 || !(files[0] instanceof Blob)) return origFetch.apply(this, arguments);
        const f = files[0];
        let key;
        try {
            const digest = await crypto.subtle.digest('SHA-1', await f.arrayBuffer());
            key = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')
                + ':' + f.size + ':' + (f.name || '');
        } catch (err) {
            return origFetch.apply(this, arguments);
        }
        const hit = cache.get(key);
        if (hit && Date.now() - hit.t < TTL) {
            const text = await hit.text;
            if (text !== null) {
                return new Response(text, { status: 200, headers: { 'content-type': 'application/json' } });
            }
        }
        const resp = origFetch.apply(this, arguments);
        const entry = { t: Date.now(), text: resp.then(r => (r.ok ? r.clone().text() : null), () => null) };
        cache.set(key, entry);
        entry.text.then(text => { if (text === null && cache.get(key) === entry) cache.delete(key); });
        while (cache.size > MAX) cache.delete(cache.keys().next().value);
        return resp;
    };
})();
// 手機大照片 / HEIC：iOS 的 canvas 有約 1677 萬像素上限，24MP/48MP 照片會讓上傳元件靜默失敗。
// 在 Gradio 收到檔案前先縮圖並轉成 JPEG：iOS 長邊 1280（ImageEditor 在 iPhone 上 1536x2048 就會讓分頁崩潰重載），其他 2048。
const MAX_SIDE = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.userAgent)) ? 1280 : 2048;
function needsShrink(f) {
    return f && /^image\/|\.(heic|heif)$/i.test(f.type || f.name) && !/gif|svg/i.test(f.type);
}
async function shrinkImage(f) {
    const url = URL.createObjectURL(f);
    try {
        const img = new Image();
        img.src = url;
        await img.decode();
        const w = img.naturalWidth, h = img.naturalHeight;
        const isHeic = /heic|heif/i.test(f.type) || /\.(heic|heif)$/i.test(f.name);
        if (Math.max(w, h) <= MAX_SIDE && !isHeic) return f;
        const k = Math.min(1, MAX_SIDE / Math.max(w, h));
        const c = document.createElement('canvas');
        c.width = Math.round(w * k); c.height = Math.round(h * k);
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
        if (!blob) return f;
        return new File([blob], f.name.replace(/\.[^.]+$/, '') + '.jpg', { type: 'image/jpeg' });
    } catch (err) {
        console.warn('shrinkImage failed, sending original', String(err));
        return f;
    } finally {
        URL.revokeObjectURL(url);
    }
}
document.addEventListener('change', async (e) => {
    const input = e.target;
    if (!(input instanceof HTMLInputElement) || input.type !== 'file') return;
    if (input.dataset.shrunk === '1') { delete input.dataset.shrunk; return; }
    const files = Array.from(input.files || []);
    if (!files.some(needsShrink)) return;
    e.stopImmediatePropagation();
    // 大照片解碼在 iPhone 上要好幾秒，沒提示會讓人以為沒反應而重複點選；顯示遮罩並擋住點擊
    const busy = document.createElement('div');
    busy.textContent = '照片處理中，請稍候…';
    busy.style.cssText = 'position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.45);color:#fff;font-size:18px;';
    document.body.appendChild(busy);
    const out = [];
    try {
        for (const f of files) out.push(needsShrink(f) ? await shrinkImage(f) : f);
    } finally {
        busy.remove();
    }
    const dt = new DataTransfer();
    out.forEach(f => dt.items.add(f));
    input.files = dt.files;
    input.dataset.shrunk = '1';
    input.dispatchEvent(new Event('change', { bubbles: true }));
}, true);
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

// 回看表格的「送到主圖 / 加到參考圖」：寫入隱藏 Textbox，再點隱藏按鈕
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.hist-send');
    if (!btn) return;
    e.preventDefault();
    const box = document.querySelector('#hist_bridge_file textarea, #hist_bridge_file input');
    const go = document.getElementById(btn.dataset.dest === 'main' ? 'hist_bridge_main' : 'hist_bridge_ref');
    if (!box || !go) return;
    box.value = btn.dataset.file;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    setTimeout(() => go.click(), 50);
});

// 佇列狀態的「取消 / 中斷」：同樣經由隱藏橋接元件
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.queue-cancel');
    if (!btn) return;
    e.preventDefault();
    const box = document.querySelector('#cancel_bridge_id textarea, #cancel_bridge_id input');
    const go = document.getElementById('cancel_bridge_btn');
    if (!box || !go) return;
    btn.disabled = true;
    box.value = btn.dataset.id;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    setTimeout(() => go.click(), 50);
});

// 佇列狀態：每 3 秒直接打 /queue_view，不用 gr.Timer。
// Gradio 每個事件的狀態變化都會向主圖 ImageEditor 重新取值，而它每取一次就把整張圖重傳一次。
let queueLastShown = null, queueBusy = false;
async function pollQueue() {
    if (queueBusy || document.hidden) return;
    const view = document.getElementById('queue_view');
    if (!view) return;
    queueBusy = true;
    try {
        const box = document.querySelector('#my_jobs_box textarea, #my_jobs_box input');
        const jobs = box ? box.value.trim() : '';
        const r = await fetch('/queue_view?jobs=' + encodeURIComponent(jobs), { cache: 'no-store' });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();
        view.innerHTML = d.html;
        if (d.newest && d.newest !== queueLastShown) {
            queueLastShown = d.newest;
            const idBox = document.querySelector('#result_bridge_id textarea, #result_bridge_id input');
            const go = document.getElementById('result_bridge_btn');
            if (idBox && go) {
                idBox.value = d.newest;
                idBox.dispatchEvent(new Event('input', { bubbles: true }));
                setTimeout(() => go.click(), 50);
            }
        }
    } catch (err) {
        view.innerHTML = '<p style="color:#b91c1c">無法讀取佇列狀態：' + String(err).replace(/[<>&]/g, '') + '</p>';
    } finally {
        queueBusy = false;
    }
}
setInterval(pollQueue, 3000);
// 按下生成後不等下一輪，送出完成（約 1–4 秒）就更新
document.addEventListener('click', (e) => {
    if (!e.target.closest('#run_btn')) return;
    [1000, 2500, 4500].forEach(ms => setTimeout(pollQueue, ms));
}, true);
document.addEventListener('visibilitychange', pollQueue);
window.addEventListener('load', () => setTimeout(pollQueue, 500));

// 過往圖片浮動視窗：捲到接近底部就自動「載入更多」（scroll 不冒泡，故用捕獲）
let pickerMoreAt = 0;
document.addEventListener('scroll', (e) => {
    const t = e.target;
    if (!(t instanceof Element) || !t.closest('#history_modal')) return;
    if (t.scrollHeight - t.scrollTop - t.clientHeight > 200) return;
    const now = Date.now();
    if (now - pickerMoreAt < 1500) return;
    pickerMoreAt = now;
    const btn = document.getElementById('picker_more_btn');
    if (btn) btn.click();
}, true);

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

class UploadLogMiddleware:
    """純 ASGI middleware，負責兩件事：
    1. 提供 GET /queue_view（佇列狀態輪詢，刻意不走 Gradio 事件）。
    2. 記錄每個上傳請求的來源、裝置與大小（docker log 的 [UPLOAD 行），診斷上傳中斷用；
       只包裝 receive 計算位元組數，不讀取或改動上傳內容。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == "/queue_view" and scope.get("method") == "GET":
            # 佇列狀態輪詢：刻意不走 Gradio 事件（理由見 build_queue_view）
            import asyncio, re, urllib.parse
            qs = urllib.parse.parse_qs(scope.get("query_string", b"").decode("latin-1"))
            jobs = [j for j in (qs.get("jobs", [""])[0]).split(",") if re.fullmatch(r"[0-9a-fA-F-]{8,64}", j)]
            html, newest = await asyncio.to_thread(build_queue_view, jobs[-50:])
            body = json.dumps({"html": html, "newest": newest}).encode("utf-8")
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json; charset=utf-8"),
                                    (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": body})
            return
        if scope["type"] != "http" or "upload" not in scope.get("path", ""):
            return await self.app(scope, receive, send)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        client = scope.get("client") or ("?", 0)
        # Tailscale Serve 轉發時來源是 127.0.0.1 並帶 X-Forwarded-For；直接連 :7860 則是手機的 tailnet IP
        if "x-forwarded-for" in headers:
            via = "tailscale-serve(https)"
        elif client[0] in ("127.0.0.1", "::1"):
            via = "localhost"
        else:
            via = "direct(http)"
        info = (f"src={client[0]} via={via} xff={headers.get('x-forwarded-for', '-')} "
                f"ts_user={headers.get('tailscale-user-login', '-')} "
                f"len={headers.get('content-length', '-')} ua={headers.get('user-agent', '-')[:120]}")
        state = {"bytes": 0, "disconnect": False, "status": None}
        t0 = time.time()

        async def counting_receive():
            msg = await receive()
            if msg["type"] == "http.request":
                state["bytes"] += len(msg.get("body", b""))
            elif msg["type"] == "http.disconnect":
                state["disconnect"] = True
            return msg

        async def status_send(msg):
            if msg["type"] == "http.response.start":
                state["status"] = msg["status"]
            await send(msg)

        print(f"[UPLOAD start] {scope['method']} {scope['path']} {info}", flush=True)
        try:
            await self.app(scope, counting_receive, status_send)
        finally:
            print(f"[UPLOAD end] status={state['status']} received={state['bytes']} "
                  f"disconnect={state['disconnect']} {time.time() - t0:.2f}s {info}", flush=True)


if __name__ == "__main__":
    from starlette.middleware import Middleware
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                head=paste_helper_script,
                allowed_paths=[COMFY_OUTPUT_DIR],
                app_kwargs={"middleware": [Middleware(UploadLogMiddleware)]})
