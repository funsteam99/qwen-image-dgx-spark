#!/usr/bin/env python3
"""常駐 Prompt Enhancer 服務 (127.0.0.1:8200)。

推論邏輯完全複用官方 prompt_rewrite/：
  - core.PROFILES      取樣參數 (t2i presence_penalty=1.5 / edit=0.0)
  - core.build_messages / core.parse_answer / core.load_image
  - run_transformers.rewrite  (含官方 PresencePenalty 與 enable_thinking=True)
本檔只負責「載入一次、常駐服務、HTTP 介面」，不改動任何官方演算法。
"""
import base64, gc, io, json, sys, time, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, "/workspace/ComfyUI/pe")
import torch
from PIL import Image
import pe_core as core
import run_transformers as rt
from transformers import AutoModelForImageTextToText, AutoProcessor

CKPTS = {
    "edit": "/workspace/ComfyUI/pe/ckpt-i2i",
    "t2i":  "/workspace/ComfyUI/pe/ckpt-t2i",
}
PROMPTS = {
    "edit": "/workspace/ComfyUI/pe/prompts/system_prompt_edit.txt",
    "t2i":  "/workspace/ComfyUI/pe/prompts/system_prompt_t2i.txt",
}
_loaded = {}
_lock = threading.Lock()


def _unload(task):
    """卸載指定 task 的模型並釋放記憶體。"""
    entry = _loaded.pop(task, None)
    if entry is None:
        return
    del entry
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"[pe] unloaded {task}", flush=True)


def get(task):
    """惰性載入，且同時只保留一個模型。

    兩個 PE 各約 19GB。若兩者並存（38GB）再加上 vLLM 與影像模型，
    會把 128GB 統一記憶體壓到 swap 全滿而使整台機器停滯（實測過）。
    故載入新模型前，先卸載另一個，讓記憶體用量固定封頂在單一模型的大小。
    代價是切換 task 時需重新載入（約 135 秒）；同一 task 連續使用不受影響。
    """
    if task in _loaded:
        return _loaded[task]
    for other in [t for t in list(_loaded) if t != task]:
        _unload(other)
    ckpt = CKPTS[task]
    if not Path(ckpt).is_dir():
        raise FileNotFoundError(f"checkpoint 尚未下載: {ckpt}")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(ckpt)
    # 與官方 run_transformers.py 一致：low_cpu_mem_usage 避免全精度 CPU 副本，
    # 再以 .to(device) 搬移單一副本 (本機記憶體吃緊，這點不可省)
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
    sysp = Path(PROMPTS[task]).read_text(encoding="utf-8")
    _loaded[task] = (model, proc, sysp, core.get_profile(task))
    print(f"[pe] loaded {task} in {time.time()-t0:.1f}s", flush=True)
    return _loaded[task]


def do_rewrite(task, user_prompt, images_b64, seed):
    model, proc, sysp, prof = get(task)
    imgs = []
    for b64 in images_b64 or []:
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        w, h = im.size
        mp = prof.image_max_pixels
        if mp and w * h > mp:                      # 同 core.load_image 的縮放規則
            s = (mp / float(w * h)) ** 0.5
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        imgs.append(im)
    msgs = core.build_messages(sysp, user_prompt, imgs)
    t0 = time.time()
    with _lock:                                    # batch size 1，序列化請求
        thinking, answer = rt.rewrite(
            model, proc, msgs,
            max_new_tokens=prof.max_new_tokens,
            temperature=prof.temperature,
            top_p=prof.top_p,
            top_k=prof.top_k,
            presence_penalty=prof.presence_penalty,
            seed=seed,
        )
    out = core.parse_answer(answer, prof)
    out["elapsed"] = round(time.time() - t0, 1)
    out["thinking_chars"] = len(thinking)
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True,
                             "loaded": sorted(_loaded),
                             "policy": "single-model: 載入新 task 前會卸載另一個",
                             "available": {k: Path(v).is_dir() for k, v in CKPTS.items()}})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/rewrite":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            task = req.get("task", "edit")
            if task not in CKPTS:
                return self._send(400, {"error": f"task 必須是 t2i 或 edit，收到 {task!r}"})
            self._send(200, do_rewrite(task, req.get("prompt", ""),
                                       req.get("images"), int(req.get("seed", 42))))
        except FileNotFoundError as e:
            self._send(503, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("[pe] serving on 127.0.0.1:8200", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8200), H).serve_forever()
