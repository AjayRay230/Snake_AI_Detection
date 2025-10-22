# /snake-ai-api/api/model_service.py

import os
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from scipy.special import softmax
<<<<<<< HEAD
import google.generativeai as genai
from dotenv import load_dotenv  # Load environment variables

# ===================== 0. LOAD ENV =====================
load_dotenv()  # loads .env file in root automatically
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found. Image generation will fail.")

# ===================== 1. PATHS & MODEL =====================
BASE_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = BASE_DIR / "ViT_PreProcessing-ops11-preprocessing-int-dynam_graph.onnx"

# Load ONNX model globally
=======
from google.generativeai import Client as GeminiClient

# ===================== 1. GLOBAL INITIALIZATION =====================

# Base directory for production-safe paths
BASE_DIR = Path(__file__).resolve().parent

# File paths
ONNX_MODEL_PATH = BASE_DIR / "ViT_PreProcessing-ops11-preprocessing-int-dynam_graph.onnx"
METADATA_PATH = BASE_DIR / "train_metadata.csv"
VENOM_DATA_PATH = BASE_DIR / "venomstatus_with_antivenom.csv"
COUNTRY_DATA_PATH = BASE_DIR / "min-train_metadata.csv"

# Load ONNX model globally (CPU only for Render)
>>>>>>> 4d071cd (Initial commit - Snake AI API)
try:
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
except FileNotFoundError:
    raise SystemExit(f"ONNX model not found at {ONNX_MODEL_PATH}. Deployment cannot proceed.")
<<<<<<< HEAD
=======

# Load CSV metadata safely
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Required CSV not found: {path}")

metadata = load_csv(METADATA_PATH)
venom_data = load_csv(VENOM_DATA_PATH)
country_metadata = load_csv(COUNTRY_DATA_PATH)

# Class names
class_names = metadata['binomial'].unique()

# Gemini API Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found. Image generation will fail.")

genai_client = GeminiClient(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
>>>>>>> 4d071cd (Initial commit - Snake AI API)

# ===================== 2. IMAGE PREPROCESSING =====================
def preprocess_image(image: Image.Image) -> np.ndarray:
<<<<<<< HEAD
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)
=======
    """
    Convert PIL image to ONNX input format (1, C, H, W)
    with normalized float32 values.
    """
    image = image.convert("RGB").resize((224, 224))
    input_array = np.array(image).astype(np.float32) / 255.0  # HWC
    input_array = np.transpose(input_array, (2, 0, 1))        # CHW
    input_array = np.expand_dims(input_array, axis=0)         # Add batch dim
    return input_array
>>>>>>> 4d071cd (Initial commit - Snake AI API)

# ===================== 3. PREDICTION FUNCTION =====================
def predict_snake(image: Image.Image) -> dict:
<<<<<<< HEAD
    try:
        # Lazy-load CSVs to save memory
        metadata = pd.read_csv(BASE_DIR / "train_metadata.csv")
        venom_data = pd.read_csv(BASE_DIR / "venomstatus_with_antivenom.csv")
        country_metadata = pd.read_csv(BASE_DIR / "min-train_metadata.csv")

        class_names = metadata['binomial'].unique()
=======
    """
    Predict snake species and return metadata (venom status, antivenom, countries, confidence)
    """
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
        # Preprocess
        input_tensor = preprocess_image(image)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Inference
        logits = session.run([output_name], {input_name: input_tensor})[0][0]
        probabilities = softmax(logits)
        predicted_class_id = int(np.argmax(probabilities))
        species_name = class_names[predicted_class_id]
        confidence = float(np.max(probabilities))

        # Venom/antivenom metadata (safe lookup)
        species_rows = venom_data[venom_data['species'] == species_name]
        species_row = species_rows.iloc[0].to_dict() if not species_rows.empty else {}

        # Country metadata lookup
        country_list = country_metadata[country_metadata['binomial'] == species_name]['country'].unique()
        countries = [c for c in country_list if c.lower() not in ('unknown', 'unidentified', 'n/a')]
        country_output = ", ".join(countries) if countries else "Distribution data unavailable"

        # Construct result
        result = {
            "species_name": species_name,
            "venom_status": int(species_row.get('venom_status', -1)),
            "antivenom_name": species_row.get('antivenom Name', 'N/A'),
            "manufacturer": species_row.get('manufacturer', 'N/A'),
            "confidence": round(confidence, 4),
            "countries": country_output
        }

        return result

    except Exception as e:
        return {"error": "Prediction failed", "detail": str(e)}
>>>>>>> 4d071cd (Initial commit - Snake AI API)

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

# ===================== 4. GEMINI IMAGE GENERATION =====================
def generate_image_from_text(prompt: str) -> tuple[bytes, Image.Image]:
<<<<<<< HEAD
    if not GEMINI_API_KEY:
        raise ConnectionError("Gemini API key not configured.")

    full_prompt = f"Highly detailed, scientifically accurate photo of a snake: {prompt}"

    # Use the official image generation model
    result = genai.models.get("imagen-3.0-generate-002").generate_content(
=======
    """
    Generate an image using Gemini API and return raw bytes and PIL image
    """
    if not genai_client:
        raise ConnectionError("Gemini API key not configured. Cannot generate image.")

    full_prompt = f"Highly detailed, scientifically accurate photo of a snake: {prompt}"

    result = genai_client.models.generate_images(
        model='imagen-3.0-generate-002',
>>>>>>> 4d071cd (Initial commit - Snake AI API)
        prompt=full_prompt,
        config=dict(
            number_of_images=1,
            output_mime_type="image/png",
            aspect_ratio="1:1"
        )
    )

    if not result.generated_images:
        raise RuntimeError("Gemini API returned no image.")

    image_bytes = result.generated_images[0].image.image_bytes
    gen_image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
<<<<<<< HEAD
    gen_image_pil = gen_image_pil.resize((224, 224))
=======

    # Optional: resize to 224x224 to match model input
    gen_image_pil = gen_image_pil.resize((224, 224))

>>>>>>> 4d071cd (Initial commit - Snake AI API)
    return image_bytes, gen_image_pil
