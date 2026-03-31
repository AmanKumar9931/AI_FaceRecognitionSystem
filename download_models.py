import os
import urllib.request

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

urls = {
    # Tiny Face Detector (RFB-320) - 2MB
    "version-RFB-320.onnx": "https://huggingface.co/infgrad/RFB-320/resolve/main/version-RFB-320.onnx",
    # MobileFaceNet (Recognition) - 4MB
    "mobilefacenet.onnx": "https://huggingface.co/XingZhang/mobilefacenet-ncnn/resolve/main/mobilefacenet.onnx"
}

print("Downloading lightweight models from Hugging Face...")

for name, url in urls.items():
    path = os.path.join('models', name)
    if not os.path.exists(path):
        print(f"⬇️ Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, path)
            print("✅ Done.")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
    else:
        print(f"✅ {name} already exists.")