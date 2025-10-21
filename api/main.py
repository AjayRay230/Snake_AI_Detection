# /snake-ai-api/api/main.py

import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # CRITICAL ADDITION
from PIL import Image

# 1. Import your core logic from the service file
from .model_service import predict_snake, generate_image_from_text 

# Initialize the FastAPI application
app = FastAPI(title="Snake AI Detection API")

# ====================================================================
# --- PRODUCTION CHANGE: ADD CORS MIDDLEWARE ---
# ====================================================================
# Define the allowed origins (replace the placeholder with your Next.js domain)
origins = [
    "http://localhost:3000",      # Next.js local development
    "http://localhost:8080",      # Another common dev port
    "http://localhost",
    "https://your-production-frontend-domain.com", # <--- **REPLACE THIS WITH YOUR LIVE NEXT.JS DOMAIN**
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # List of allowed origins
    allow_credentials=True,       
    allow_methods=["*"],          # Allow POST, GET, etc.
    allow_headers=["*"],          
)
# ====================================================================


# ====================================================================
# Endpoint 1: Image Upload and Prediction 
# ====================================================================

@app.post("/predict-image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """Accepts an image file upload and returns the snake identification results."""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        prediction_result = predict_snake(image)
        
        return JSONResponse(content=prediction_result)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return JSONResponse(content={"error": "An error occurred during image prediction.", "detail": str(e)}, status_code=500)

# ====================================================================
# Endpoint 2: Text Description, Image Generation, and Prediction 
# ====================================================================

@app.post("/generate-and-predict")
async def generate_and_predict_endpoint(description: str = Form(...)):
    """Generates an image from text, runs prediction, and returns the image (Base64) and results."""
    try:
        image_bytes, gen_image_pil = generate_image_from_text(description)
        prediction_result = predict_snake(gen_image_pil)
        
        # Encode the generated image to Base64 for display
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response_data = {
            "generated_image_base64": base64_image,
            "prediction": prediction_result
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Generation/Prediction Error: {e}")
        return JSONResponse(content={"error": "An error occurred during image generation or prediction.", "detail": str(e)}, status_code=500)