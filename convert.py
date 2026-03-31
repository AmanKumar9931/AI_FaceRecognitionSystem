import torch
import torch.onnx
from models.mobilefacenet import MobileFaceNet  # You need the model definition for this

# 1. Load your .pth file
model = MobileFaceNet() 
model.load_state_dict(torch.load('models/mobilefacenet_model_best.pth', map_location='cpu'))
model.eval()

# 2. Create dummy input (standard face size)
dummy_input = torch.randn(1, 3, 112, 112)

# 3. Export to ONNX
torch.onnx.export(model, dummy_input, "models/mobilefacenet.onnx", input_names=['input'], output_names=['output'])
print("✅ Converted to models/mobilefacenet.onnx")