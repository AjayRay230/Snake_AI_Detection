# /snake-ai-api/api/main.py

import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import optimized service
from .model_service import predict_snake, generate_image_from_text

# Initialize FastAPI
app = FastAPI(title="Snake AI Detection API")

# ===================== 1. CORS MIDDLEWARE =====================
# Allowed origins (update with your frontend domain)
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost",
    "https://your-production-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===================== 2. IMAGE UPLOAD & PREDICTION =====================
@app.post("/predict-image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image file upload and returns snake prediction metadata.
    """
    try:
        image_data = await file.read()

        # Load image safely in RGB
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Run prediction (uses global ONNX session)
        prediction_result = predict_snake(image)

        return JSONResponse(content=prediction_result)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return JSONResponse(
            content={"error": "An error occurred during image prediction.", "detail": str(e)},
            status_code=500
        )

# ===================== 3. TEXT-TO-IMAGE & PREDICTION =====================
@app.post("/generate-and-predict")
async def generate_and_predict_endpoint(description: str = Form(...)):
    """
    Generates an image from text description, runs snake prediction, and returns
    both the Base64 image and prediction metadata.
    """
    try:
        # Generate image using Gemini API
        image_bytes, gen_image_pil = generate_image_from_text(description)

        # Run snake prediction on generated image
        prediction_result = predict_snake(gen_image_pil)

        # Encode generated image to Base64 for frontend display
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        response_data = {
            "generated_image_base64": base64_image,
            "prediction": prediction_result
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Generation/Prediction Error: {e}")
        return JSONResponse(
            content={"error": "An error occurred during image generation or prediction.", "detail": str(e)},
            status_code=500
        )
