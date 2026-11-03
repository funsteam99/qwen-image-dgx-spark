"""qwen-image-comfyui 按需啟動 / 閒置關閉（主機上執行，不在容器內）。

前端（qwen-gradio-ui）常駐；ComfyUI 平常停著，有人按「開始生成」才由這裡啟動，
閒置一段時間、佇列也空了就自動停掉，把記憶體讓給 Spark 上其他專案。

和其他專案的約定沿用 minimax-h3-studio 已部署的做法，不另立一套：
  - 對方跑 H3 前打 7860 的 POST /busy_status 設 busy=true，並自己停掉 qwen-image-comfyui；
    結束（含被殺）時清回 false。旗標存在 ~/qwen-image/busy_status.json。
  - 這裡只在 busy=false、沒有 h3-/demucs 容器、可用記憶體夠時才啟動；
    啟動後再看一次旗標，避免和對方「剛設 busy、還沒停容器」的瞬間撞在一起。
  - 本服務從不停其他專案的東西（8002、8006 等），只管 qwen-image-comfyui。

只聽 127.0.0.1，給同機 network_mode=host 的前端呼叫：
  GET  /status  → {"state": "running|stopped|starting", "busy": bool, "reason": str|null, ...}
  POST /ensure  → 確保 ComfyUI 在跑並就緒；回 {"ok": true} 或 {"ok": false, "reason": "..."}
  POST /release → 佇列空才停掉 ComfyUI；回 {"ok": true} 或 {"ok": false, "reason": "busy"}

以 systemd --user 常駐（見 comfy-ondemand.service）。
"""
import json
import os
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = ("127.0.0.1", int(os.environ.get("ONDEMAND_PORT", "8190")))
CONTAINER = "qwen-image-comfyui"
COMFY = "http://127.0.0.1:8188"
QWEN_DIR = os.path.expanduser(os.environ.get("QWEN_IMAGE_DIR", "~/qwen-image"))
BUSY_STATUS_FILE = os.path.join(QWEN_DIR, "busy_status.json")
# Qwen-Image 2.1 int8 + 文字編碼器實測常駐約 25~30GB，bf16 再多約 14GB；留餘裕給系統與 8002。
MIN_AVAILABLE_GB = float(os.environ.get("MIN_AVAILABLE_GB", "45"))
IDLE_STOP_MIN = float(os.environ.get("IDLE_STOP_MIN", "15"))
READY_TIMEOUT_S = 240

STATE_LOCK = threading.Lock()   # 啟動 / 停止互斥，同一時間只有一個人在動容器
starting = False
last_active = time.time()


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def docker(*args, timeout=60):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def container_running(name=CONTAINER):
    r = docker("inspect", "-f", "{{.State.Running}}", name)
    return r.returncode == 0 and r.stdout.strip() == "true"


def other_gpu_jobs():
    """其他專案的重任務（h3-、demucs 容器）。對方腳本自己也用同樣的名稱規則做互斥。"""
    r = docker("ps", "--format", "{{.Names}}")
    return [n for n in r.stdout.split() if n.startswith(("h3-", "demucs"))]


def busy_status():
    try:
        with open(BUSY_STATUS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return bool(d.get("busy")), d.get("reason") or None
    except Exception:
        return False, None


def mem_available_gb():
    # Spark 是 CPU/GPU 共用記憶體，看主機 MemAvailable 才準
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def comfy_queue():
    """回傳 (執行中+排隊數)；連不上回 None。"""
    try:
        with urllib.request.urlopen(f"{COMFY}/queue", timeout=5) as r:
            q = json.load(r)
        return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
    except Exception:
        return None


def comfy_ready():
    try:
        with urllib.request.urlopen(f"{COMFY}/system_stats", timeout=5):
            return True
    except Exception:
        return False


def blocked_reason():
    busy, reason = busy_status()
    if busy:
        return reason or "其他專案正在使用 GPU"
    others = other_gpu_jobs()
    if others:
        return f"其他專案正在使用 GPU（{', '.join(others)}）"
    return None


def ensure():
    """確保 ComfyUI 在跑且就緒。回 (ok, reason)。"""
    global starting, last_active
    last_active = time.time()
    if container_running() and comfy_ready():
        return True, None
    with STATE_LOCK:
        if not container_running():
            why = blocked_reason()
            if why:
                return False, why
            avail = mem_available_gb()
            if avail < MIN_AVAILABLE_GB:
                return False, f"可用記憶體不足（剩 {avail:.0f} GB，需要 {MIN_AVAILABLE_GB:.0f} GB）"
            log(f"啟動 {CONTAINER}（可用 {avail:.0f} GB）")
            starting = True
            try:
                r = docker("start", CONTAINER)
                if r.returncode != 0:
                    return False, f"docker start 失敗：{r.stderr.strip()[:200]}"
                # 啟動前檢查到啟動完成之間，對方可能剛設了 busy：讓出去
                why = blocked_reason()
                if why:
                    log(f"啟動後發現其他專案要用 GPU，停回去：{why}")
                    docker("stop", CONTAINER)
                    return False, why
            finally:
                starting = False
        deadline = time.time() + READY_TIMEOUT_S
        starting = True
        try:
            while time.time() < deadline:
                if comfy_ready():
                    last_active = time.time()
                    return True, None
                if not container_running():
                    return False, "ComfyUI 容器啟動後又結束了，請看 docker logs"
                time.sleep(2)
        finally:
            starting = False
    return False, f"ComfyUI {READY_TIMEOUT_S} 秒內沒有就緒"


def release(reason):
    """佇列空才停。回 (ok, reason)。"""
    with STATE_LOCK:
        if not container_running():
            return True, None
        n = comfy_queue()
        if n:
            return False, "busy"
        log(f"停止 {CONTAINER}：{reason}")
        docker("stop", CONTAINER, timeout=120)
        return True, None


def idle_watchdog():
    global last_active
    while True:
        time.sleep(30)
        try:
            if starting or not container_running():
                continue
            if comfy_queue():
                last_active = time.time()
                continue
            if time.time() - last_active >= IDLE_STOP_MIN * 60:
                release(f"閒置 {IDLE_STOP_MIN:.0f} 分鐘")
        except Exception as e:
            log(f"watchdog: {type(e).__name__}: {e}")


def status():
    busy, reason = busy_status()
    if starting:
        state = "starting"
    elif container_running():
        state = "running"
    else:
        state = "stopped"
    return {"state": state, "busy": busy, "reason": reason,
            "idle_stop_min": IDLE_STOP_MIN,
            "mem_available_gb": round(mem_available_gb(), 1)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            return self._send(200, status())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/ensure":
            ok, reason = ensure()
            return self._send(200, {"ok": ok, "reason": reason})
        if self.path == "/release":
            ok, reason = release("外部要求釋放")
            return self._send(200, {"ok": ok, "reason": reason})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=idle_watchdog, daemon=True).start()
    log(f"comfy-ondemand 聽 {LISTEN[0]}:{LISTEN[1]}，閒置 {IDLE_STOP_MIN:.0f} 分鐘自動停止")
    ThreadingHTTPServer(LISTEN, Handler).serve_forever()
