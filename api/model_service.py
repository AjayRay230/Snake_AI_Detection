# /snake-ai-api/api/model_service.py

import os
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from scipy.special import softmax
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
try:
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
except FileNotFoundError:
    raise SystemExit(f"ONNX model not found at {ONNX_MODEL_PATH}. Deployment cannot proceed.")

# ===================== 2. IMAGE PREPROCESSING =====================
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)

# ===================== 3. PREDICTION FUNCTION =====================
def predict_snake(image: Image.Image) -> dict:
    try:
        # Lazy-load CSVs to save memory
        metadata = pd.read_csv(BASE_DIR / "train_metadata.csv")
        venom_data = pd.read_csv(BASE_DIR / "venomstatus_with_antivenom.csv")
        country_metadata = pd.read_csv(BASE_DIR / "min-train_metadata.csv")

        class_names = metadata['binomial'].unique()

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
    if not GEMINI_API_KEY:
        raise ConnectionError("Gemini API key not configured.")

    full_prompt = f"Highly detailed, scientifically accurate photo of a snake: {prompt}"

    # Use the official image generation model
    result = genai.models.get("imagen-3.0-generate-002").generate_content(
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
    gen_image_pil = gen_image_pil.resize((224, 224))
    return image_bytes, gen_image_pil
