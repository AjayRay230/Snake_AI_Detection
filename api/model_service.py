# /snake-ai-api/api/model_service.py

import os
import io
import pandas as pd
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch # Included because torchvision requires it, though only transforms are used here
from torchvision import transforms
from scipy.special import softmax
import google.generativeai as genai
from pathlib import Path # CRITICAL for production-ready file path handling

# ===================== 1. GLOBAL INITIALIZATION (Production Ready) =====================

# Define the base directory (the 'api' folder)
BASE_DIR = Path(__file__).resolve().parent

# --- File Paths (Constructed using BASE_DIR) ---
ONNX_MODEL_PATH = BASE_DIR / "ViT_PreProcessing-ops11-preprocessing-int-dynam_graph.onnx"
METADATA_PATH = BASE_DIR / "train_metadata.csv"
VENOM_DATA_PATH = BASE_DIR / "venomstatus_with_antivenom.csv"
# We use min-train_metadata.csv for country data lookup
COUNTRY_DATA_PATH = BASE_DIR / "min-train_metadata.csv" 

try:
    # 1.1 Load ONNX Model
    # Convert Path object to string for onnxruntime
    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    
    # 1.2 Load Metadata and Data
    metadata = pd.read_csv(METADATA_PATH)
    venom_data = pd.read_csv(VENOM_DATA_PATH)
    country_metadata = pd.read_csv(COUNTRY_DATA_PATH) 

    # Extract class names (binomial)
    class_names = metadata['binomial'].unique()

except FileNotFoundError as e:
    # Stop the application if essential assets are missing in the production environment
    print(f"FATAL ERROR: Essential data file not found at deployment: {e.filename}")
    # The Uvicorn server (run by FastAPI) will likely catch this and stop.
    raise SystemExit(f"Deployment failed due to missing asset: {e.filename}. Check Dockerfile COPY commands.")
    
# 1.3 Initialize Gemini API Client (Securely using Environment Variable)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

if not GEMINI_API_KEY:
     print("WARNING: GEMINI_API_KEY not found. Image generation will fail.")

genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ===================== 2. IMAGE PREPROCESSING =====================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Applies the necessary transformations to a PIL Image 
    to prepare it for the ONNX model input.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image = transform(image)
    
    # Add a batch dimension (1, C, H, W) and convert to float32 NumPy array
    input_tensor = np.expand_dims(image.numpy(), axis=0).astype(np.float32)
    return input_tensor


# ===================== 3. PREDICTION FUNCTION =====================

def predict_snake(image: Image.Image) -> dict:
    """
    Runs the prediction pipeline and looks up all associated metadata 
    (venom status, antidote, country).
    """
    if not session:
        return {"error": "Model not loaded.", "species_name": "Unknown", "venom_status": -1, "antivenom_name": "N/A", "manufacturer": "N/A", "confidence": 0.0, "countries": "N/A"}
        
    # --- Inference Logic ---
    input_tensor = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run ONNX prediction
    logits = session.run([output_name], {input_name: input_tensor})[0][0]
    
    # Post-processing (Softmax and Argmax)
    probabilities = softmax(logits)
    predicted_class_id = np.argmax(probabilities)
    species_name = class_names[predicted_class_id]
    confidence = float(np.max(probabilities))
    
    # --- Metadata Lookup ---
    # 3.4 Venom/Antivenom Lookup
    species_row = venom_data[venom_data['species'] == species_name].iloc[0]
    
    # 3.4 Country Lookup (using min-train_metadata.csv)
    # Filter the country metadata for the predicted species name
    country_list = country_metadata[country_metadata['binomial'] == species_name]['country'].unique()
    
    # Clean the list and convert to a comma-separated string
    countries = [c for c in country_list if c.lower() not in ('unknown', 'unidentified', 'n/a')]
    country_output = ", ".join(countries) if countries else "Distribution data unavailable"
    
    # 3.5 Format Result
    result = {
        "species_name": species_name,
        "venom_status": int(species_row['venom_status']), 
        "antivenom_name": species_row.get('antivenom Name', 'N/A'),
        "manufacturer": species_row.get('manufacturer', 'N/A'),
        "confidence": round(confidence, 4), 
        "countries": country_output 
    }
    return result


# ===================== 4. IMAGE GENERATION FUNCTION =====================

def generate_image_from_text(prompt: str) -> tuple[bytes, Image.Image]:
    """
    Uses the Gemini API (Imagen 3.0) to generate an image from a text description.
    
    Returns:
        A tuple: (raw JPEG bytes of the generated image, PIL Image object)
    """
    
    if not GEMINI_API_KEY:
        # Raise a specific error if the endpoint relies on the key
        raise ConnectionError("Gemini API key not configured. Cannot generate image.")

    # Add context to the prompt
    full_prompt = f"Highly detailed, scientifically accurate photo of a snake matching this description: {prompt}"
    
    # Call the image generation model
    result = genai_client.models.generate_images(
        model='imagen-3.0-generate-002',
        prompt=full_prompt,
        config=dict(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="1:1"
        )
    )
    
    if not result.generated_images:
        raise RuntimeError("Gemini API returned no image.")

    # Decode bytes
    image_bytes = result.generated_images[0].image.image_bytes
    
    # Create PIL Image object for the predict_snake function
    gen_image_pil = Image.open(io.BytesIO(image_bytes))
    
    return image_bytes, gen_image_pil