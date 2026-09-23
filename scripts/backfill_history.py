#!/usr/bin/env python3
"""從 ComfyUI /history 回填生成紀錄。

「原始提示」無法回填 —— 改寫前的短句只存在於當時的瀏覽器，未曾留存。
僅回填 WebUI 產生的 QwenImage21_Out_*，排除開發測試用的輸出。
"""
import json, os, struct, time, urllib.request

H = "127.0.0.1:8188"
OUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", os.path.expanduser("~/qwen-image/ComfyUI/output"))
LOG = os.environ.get("HISTORY_LOG", os.path.expanduser("~/qwen-image/generation_history.jsonl"))
PREFIX = "QwenImage21_Out_"

def png_size(path):
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return f"{w}x{h}"
    except Exception:
        pass
    return ""

existing = set()
if os.path.isfile(LOG):
    for line in open(LOG, encoding="utf-8"):
        try:
            existing.add(json.loads(line).get("file"))
        except Exception:
            pass

hist = json.loads(urllib.request.urlopen(f"http://{H}/history?max_items=200").read())
rows = []
for pid, v in hist.items():
    files = [im["filename"] for o in v.get("outputs", {}).values()
             for im in o.get("images", [])]
    files = [f for f in files if f.startswith(PREFIX)]
    if not files:
        continue
    fn = files[0]
    if fn in existing or not os.path.isfile(os.path.join(OUT_DIR, fn)):
        continue
    p = v["prompt"][2]
    enc = p.get("4", {}).get("inputs", {})
    ks = p.get("6", {}).get("inputs", {})
    nimg = len([k for k in enc if k.startswith("image")])
    mode = "t2i" if nimg == 0 else ("multi" if nimg > 2 else "edit")
    ts = ""
    for m in v.get("status", {}).get("messages", []):
        if m[0] == "execution_start":
            ts = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(m[1]["timestamp"] / 1000))
            break
    rows.append({
        "ts": ts, "original": "", "final": enc.get("prompt", ""),
        "file": fn, "mode": mode, "size": png_size(os.path.join(OUT_DIR, fn)),
        "seed": ks.get("seed"), "steps": ks.get("steps"),
        "precision": p.get("1", {}).get("inputs", {}).get("unet_name", ""),
        "backfilled": True,
    })

rows.sort(key=lambda r: r["ts"] or "")
with open(LOG, "a", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"回填 {len(rows)} 筆")
for r in rows:
    print(" ", r["ts"], r["file"], r["mode"], r["size"], "| prompt:", (r["final"] or "")[:40])
