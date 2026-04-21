import os
import base64
import numpy as np
import cv2
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from PIL import Image
from io import BytesIO
import onnxruntime as ort
import gc

app = Flask(__name__)

# --- CONFIGURATION ---
DATASET_FOLDER = 'dataset'
CSV_FILE = 'attendance.csv'
MODELS_FOLDER = 'models'
os.makedirs(DATASET_FOLDER, exist_ok=True)

# --- GLOBAL VARS ---
known_embeddings = []
known_reg_nos = []
known_names = []

# --- AI ENGINE (ONNX) ---
# Load models once at startup
detector_path = os.path.join(MODELS_FOLDER, 'version-RFB-320.onnx')
recognizer_path = os.path.join(MODELS_FOLDER, 'MobileNet-v2.onnx')  # <--- UPDATED FILENAME

print(f"Loading models: {detector_path} and {recognizer_path}...")

try:
    # Initialize ONNX Sessions (Lightweight!)
    ort_det = ort.InferenceSession(detector_path)
    ort_rec = ort.InferenceSession(recognizer_path)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error loading models: {e}")
    # We don't exit here so the app can still start, but it won't work until fixed.

# --- HELPER: IMAGE PREPROCESSING ---
def preprocess_image(image):
    """
    Converts a PIL image to the format the AI expects.
    MobileNet-v2 standard input is 224x224.
    """
    # 1. Convert PIL to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 2. Resize to 224x224 (Standard for MobileNetV2)
    img = cv2.resize(img, (224, 224))

    # 3. Normalize (mathematics to make colors -1 to 1)
    img = (img - 127.5) / 128.0

    # 4. Transpose (HWC -> CHW) - AI expects Color channels first
    img = np.transpose(img, (2, 0, 1))

    # 5. Add Batch Dimension (1, 3, 224, 224)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    return img

def get_embedding(image):
    """
    Runs the image through the ONNX model to get numbers.
    """
    input_tensor = preprocess_image(image)
    
    # Run Inference
    # We automatically get the input name so it works with any model version
    input_name = ort_rec.get_inputs()[0].name
    embedding = ort_rec.run(None, {input_name: input_tensor})[0]
    
    # Flatten and Normalize the vector
    embedding = embedding.flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.json
        name = data.get('name')
        reg_no = data.get('reg_no')
        images_b64 = data.get('images')

        if not name or not reg_no or not images_b64:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        print(f"📝 Registering {name}...")
        student_folder = os.path.join(DATASET_FOLDER, reg_no)
        os.makedirs(student_folder, exist_ok=True)

        embeddings = []
        # Limit to 5 images to keep RAM low
        images_to_use = images_b64[:5]

        for i, img_str in enumerate(images_to_use):
            try:
                if ',' in img_str: header, encoded = img_str.split(",", 1)
                else: encoded = img_str
                
                img_data = base64.b64decode(encoded)
                image = Image.open(BytesIO(img_data)).convert('RGB')
                
                # Get Embedding (Using our lightweight ONNX function)
                emb = get_embedding(image)
                embeddings.append(emb)

                # Memory Cleanup
                del image, img_data, emb
                gc.collect()

            except Exception as e:
                print(f"⚠️ Skipped image {i}: {e}")

        if not embeddings:
            return jsonify({"status": "error", "message": "No faces processed"}), 400

        # Save Average
        master_embedding = np.mean(embeddings, axis=0)
        np.save(os.path.join(student_folder, "embedding.npy"), master_embedding)
        
        with open(os.path.join(student_folder, "info.txt"), "w") as f:
            f.write(f"{name},{reg_no}")

        load_known_faces()
        return jsonify({"status": "success", "message": f"Registered {name}!"})

    except Exception as e:
        print(f"Register Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/mark_attendance', methods=['POST'])
def api_mark_attendance():
    global known_embeddings, known_names, known_reg_nos
    data = request.json
    image_data = data.get('image')

    if not image_data: return jsonify({"status": "error"}), 400

    try:
        if ',' in image_data: header, encoded = image_data.split(",", 1)
        else: encoded = image_data
        
        img_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')

        # Get live embedding
        current_emb = get_embedding(image)

        # Match against database
        if len(known_embeddings) == 0:
            return jsonify({"match": False})

        dist_list = []
        for known_emb in known_embeddings:
            # Cosine Distance
            dist = np.dot(current_emb, known_emb)
            dist_list.append(dist)
        
        # Find highest similarity (closer to 1.0 is better)
        max_score = max(dist_list)
        max_index = dist_list.index(max_score)

        # Threshold: 0.5 is usually good for MobileNet
        if max_score > 0.5:
            name = known_names[max_index]
            reg_no = known_reg_nos[max_index]
            mark_attendance_csv(name, reg_no)
            return jsonify({"match": True, "student": name})
        else:
            return jsonify({"match": False})

    except Exception as e:
        print(f"Attendance Error: {e}")
        return jsonify({"status": "error"}), 500

# --- DATA MANAGEMENT ---
def load_known_faces():
    global known_embeddings, known_reg_nos, known_names
    known_embeddings, known_reg_nos, known_names = [], [], []

    if not os.path.exists(DATASET_FOLDER): return

    for reg_no in os.listdir(DATASET_FOLDER):
        path = os.path.join(DATASET_FOLDER, reg_no)
        if os.path.isdir(path):
            try:
                emb = np.load(os.path.join(path, "embedding.npy"))
                with open(os.path.join(path, "info.txt"), "r") as f:
                    name = f.read().split(',')[0]
                known_embeddings.append(emb)
                known_reg_nos.append(reg_no)
                known_names.append(name)
            except: pass
    print(f"✅ Loaded {len(known_embeddings)} students.")

def mark_attendance_csv(name, reg_no):
    now = datetime.now()
    date_str, time_str = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')
    
    if not os.path.isfile(CSV_FILE):
        df = pd.DataFrame(columns=['Date', 'Time', 'Name', 'RegNo', 'Status'])
        df.to_csv(CSV_FILE, index=False)
    
    df = pd.read_csv(CSV_FILE)
    if df[(df['RegNo'] == str(reg_no)) & (df['Date'] == date_str)].empty:
        new_row = pd.DataFrame([{'Date': date_str, 'Time': time_str, 'Name': name, 'RegNo': str(reg_no), 'Status': 'Present'}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)