# /snake-ai-api/api/model_service.py

import os
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from scipy.special import softmax
import google.generativeai as genai  # updated import

# ===================== 1. GLOBAL INITIALIZATION =====================

BASE_DIR = Path(__file__).resolve().parent

ONNX_MODEL_PATH = BASE_DIR / "ViT_PreProcessing-ops11-preprocessing-int-dynam_graph.onnx"
METADATA_PATH = BASE_DIR / "train_metadata.csv"
VENOM_DATA_PATH = BASE_DIR / "venomstatus_with_antivenom.csv"
COUNTRY_DATA_PATH = BASE_DIR / "min-train_metadata.csv"

# Load ONNX model globally
try:
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
except FileNotFoundError:
    raise SystemExit(f"ONNX model not found at {ONNX_MODEL_PATH}. Deployment cannot proceed.")

# Load CSV metadata safely
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Required CSV not found: {path}")

metadata = load_csv(METADATA_PATH)
venom_data = load_csv(VENOM_DATA_PATH)
country_metadata = load_csv(COUNTRY_DATA_PATH)

class_names = metadata['binomial'].unique()

# ===================== 2. GEMINI API CONFIG =====================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found. Image generation will fail.")

# ===================== 3. IMAGE PREPROCESSING =====================

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    input_array = np.array(image).astype(np.float32) / 255.0  # HWC
    input_array = np.transpose(input_array, (2, 0, 1))        # CHW
    input_array = np.expand_dims(input_array, axis=0)         # Add batch dim
    return input_array

# ===================== 4. PREDICTION FUNCTION =====================

def predict_snake(image: Image.Image) -> dict:
    if not session:
        return {
            "error": "Model not loaded.",
            "species_name": "Unknown",
            "venom_status": -1,
            "antivenom_name": "N/A",
            "manufacturer": "N/A",
            "confidence": 0.0,
            "countries": "N/A"
        }

    try:
        input_tensor = preprocess_image(image)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        logits = session.run([output_name], {input_name: input_tensor})[0][0]
        probabilities = softmax(logits)
        predicted_class_id = int(np.argmax(probabilities))
        species_name = class_names[predicted_class_id]
        confidence = float(np.max(probabilities))

        species_rows = venom_data[venom_data['species'] == species_name]
        species_row = species_rows.iloc[0].to_dict() if not species_rows.empty else {}

        country_list = country_metadata[country_metadata['binomial'] == species_name]['country'].unique()
        countries = [c for c in country_list if c.lower() not in ('unknown', 'unidentified', 'n/a')]
        country_output = ", ".join(countries) if countries else "Distribution data unavailable"

        return {
            "species_name": species_name,
            "venom_status": int(species_row.get('venom_status', -1)),
            "antivenom_name": species_row.get('antivenom Name', 'N/A'),
            "manufacturer": species_row.get('manufacturer', 'N/A'),
            "confidence": round(confidence, 4),
            "countries": country_output
        }

    except Exception as e:
        return {"error": "Prediction failed", "detail": str(e)}

# ===================== 5. IMAGE GENERATION FUNCTION =====================

def generate_image_from_text(prompt: str) -> tuple[bytes, Image.Image]:
    if not GEMINI_API_KEY:
        raise ConnectionError("Gemini API key not configured. Cannot generate image.")

    full_prompt = f"Highly detailed, scientifically accurate photo of a snake: {prompt}"

    result = genai.models.get("models/gemini-2.5-flash-preview-05-20").generate_content(
        prompt=full_prompt,
        temperature=1.0,
        max_output_tokens=1024
    )

    if not result or not result.output_text:
        raise RuntimeError("Gemini API returned no image or text.")

    # If your Gemini plan supports image output:
    # Use the proper image generation model
    # e.g., genai.models.get("imagen-3.0-generate-002").generate_content(...)

    # For now, return the text as a placeholder
    image_bytes = result.output_text.encode("utf-8")
    gen_image_pil = Image.new("RGB", (224, 224), color=(255, 255, 255))
    return image_bytes, gen_image_pil
