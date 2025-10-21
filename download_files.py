import os
import gdown  # pip install gdown

# Ensure the "api" directory exists
os.makedirs("api", exist_ok=True)

# ✅ Replace these with your actual Google Drive FILE IDs
MODEL_ID = "1HFCe30fz0AGfUrvXVkyblEnJuN5nzzLH"
CSV_ID = "18Z7GlbtA-PULXYa6t9KVvhYr6Hv-T5uj"
XLSX_ID = "1aj0so1m7LV7b7Aqr1W7Q_JrmsAPSyLr1"
Train_ID = "1HnX_p18IrNZtwL0hcKz_MIOJM4KbzzP3"
# Convert them into downloadable URLs
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
CSV_URL = f"https://drive.google.com/uc?id={CSV_ID}"
XLSX_URL = f"https://drive.google.com/uc?id={XLSX_ID}"
Train_URL = f"https://drive.google.com/uc?id={Train_ID}"

# Local file paths
MODEL_PATH = "api/ViT_PreProcessing-ops11-preprocessing-int-dynam_graph.onnx"
CSV_PATH = "api/venomstatus_with_antivenom.csv"
XLSX_PATH = "api/venomstatus_with_antivenom.xlsx"
Train_Path="api/train_metadata.csv"
def download_file(url, path):
    """Downloads a file from Google Drive if it doesn't already exist."""
    if not os.path.exists(path):
        print(f"⬇️ Downloading {os.path.basename(path)}...")
        gdown.download(url, path, quiet=False)
        print(f"✅ Saved to {path}")
    else:
        print(f"✅ {os.path.basename(path)} already exists.")

# Run downloads
download_file(MODEL_URL, MODEL_PATH)
download_file(CSV_URL, CSV_PATH)
download_file(XLSX_URL, XLSX_PATH)
download_file(Train_URL,Train_Path)
