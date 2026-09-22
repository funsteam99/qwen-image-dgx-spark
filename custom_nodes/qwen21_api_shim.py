"""TextEncodeQwenImage21API — 純傳輸層轉接節點。

ComfyUI 0.37.0 的 autogrow 動態輸入(COMFY_AUTOGROW_V3)無法透過 /prompt API 傳值：
平鋪 image_1 會 TypeError，巢狀 images 會被靜默丟棄。此節點只把輸入改成普通的
image1..image10，內部直接呼叫官方 TextEncodeQwenImage21.execute()，不改任何演算法。
"""
from comfy_extras.nodes_qwen import TextEncodeQwenImage21

MAX_IMAGES = 10


class TextEncodeQwenImage21API:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {"vae": ("VAE",)}
        for i in range(1, MAX_IMAGES + 1):
            opt[f"image{i}"] = ("IMAGE",)
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/qwen image"

    def execute(self, clip, prompt, negative_prompt, resolution=1024, vae=None, **kwargs):
        images = {}
        for i in range(1, MAX_IMAGES + 1):
            img = kwargs.get(f"image{i}")
            if img is not None:
                images[f"image_{i}"] = img
        out = TextEncodeQwenImage21.execute(
            clip=clip, prompt=prompt, negative_prompt=negative_prompt,
            vae=vae, resolution=resolution, images=images,
        )
        for attr in ("result", "_result", "values"):
            v = getattr(out, attr, None)
            if isinstance(v, (tuple, list)):
                return tuple(v)
        return tuple(out)


NODE_CLASS_MAPPINGS = {"TextEncodeQwenImage21API": TextEncodeQwenImage21API}
NODE_DISPLAY_NAME_MAPPINGS = {"TextEncodeQwenImage21API": "Text Encode Qwen Image 2.1 (API)"}
