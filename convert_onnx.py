import os
import requests
from ultralytics import YOLO

MODEL_URL = os.getenv("MODEL_URL", "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt?download=true")
MODEL_PATH = "doclayout_yolo_ft.pt"
ONNX_PATH = "doclayout_yolo_ft.onnx"

def download_file(url, dest):
    print(f"📥 Downloading model from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Model saved to {dest}")

def main():
    # 1️⃣ 下载模型
    if not os.path.exists(MODEL_PATH):
        download_file(MODEL_URL, MODEL_PATH)

    # 2️⃣ 导出ONNX
    print("🚀 Exporting to ONNX...")
    model = YOLO(MODEL_PATH)
    model.export(
        format="onnx",
        # opset=12,
        # simplify=True,
        # nms=False
    )
    print(f"✅ Export complete: {ONNX_PATH}")

if __name__ == "__main__":
    main()

