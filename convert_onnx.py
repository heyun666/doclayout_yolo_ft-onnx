import os
import sys
import traceback
import requests
import torch
from ultralytics import YOLO

# ✅ 尝试导入 doclayout_yolo 模块（若已安装）
try:
    import doclayout_yolo.nn.modules.g2l_crm as g2l_crm
except Exception as e:
    print("⚠️ Warning: doclayout_yolo not found, attempting to install...")
    os.system("pip install doclayout-yolo -U")
    import doclayout_yolo.nn.modules.g2l_crm as g2l_crm

MODEL_URL = os.getenv("MODEL_URL", "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt?download=true")
MODEL_PATH = "doclayout_yolo_ft.pt"
ONNX_PATH = "doclayout_yolo_ft.onnx"


def download_model():
    """Download model from URL if not already present."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists: {MODEL_PATH}")
        return

    if not MODEL_URL:
        print("❌ MODEL_URL environment variable not set.")
        sys.exit(1)

    print(f"📥 Downloading model from {MODEL_URL} ...")
    r = requests.get(MODEL_URL, stream=True)
    if r.status_code != 200:
        print(f"❌ Failed to download model: {r.status_code}")
        sys.exit(1)

    total = int(r.headers.get("content-length", 0))
    with open(MODEL_PATH, "wb") as f:
        downloaded = 0
        for chunk in r.iter_content(8192):
            f.write(chunk)
            downloaded += len(chunk)
            percent = downloaded / total * 100 if total else 0
            print(f"\r📦 Downloaded {downloaded / 1024 / 1024:.1f} MB ({percent:.1f}%)", end="")
    print(f"\n✅ Model saved to {MODEL_PATH}")


def patch_dilated_conv():
    """Patch DilatedBlock to avoid accessing Conv.bn directly during ONNX export."""

    def safe_dilated_conv(self, x, dilation):
        # 使用 self.dcv 直接推理，而不访问其内部 bn
        padding = dilation * (self.k // 2)
        self.dcv.conv.dilation = (dilation, dilation)
        self.dcv.conv.padding = (padding, padding)
        return self.dcv(x)

    # 替换原函数
    g2l_crm.DilatedBlock.dilated_conv = safe_dilated_conv
    print("🔧 Patched DilatedBlock.dilated_conv() for safe ONNX export.")


def main():
    try:
        download_model()
        patch_dilated_conv()

        print("🚀 Exporting to ONNX...")

        model = YOLO(MODEL_PATH)
        model.export(
            format="onnx",
            opset=13,         # ✅ 降为 13，稳定支持
            simplify=False,   # 防止 onnxsim 改坏图
            #opset=12,
            #simplify=True,
            #imgsz=640,
            #dynamic=True,
            #nms=False
        )

        print(f"✅ ONNX export complete: {ONNX_PATH}")

    except Exception as e:
        print("❌ Export failed with error:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

