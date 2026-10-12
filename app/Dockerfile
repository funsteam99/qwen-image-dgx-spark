FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /workspace/ComfyUI

COPY ComfyUI/requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    pip uninstall -y torchaudio && \
    pip install --no-cache-dir gradio pillow-heif

EXPOSE 8188 7860

CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--highvram"]
