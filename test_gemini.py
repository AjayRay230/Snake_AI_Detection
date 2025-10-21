import os
import google.generativeai as genai

# 1️⃣ Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAvmKaVC7n67ybRRJXb32p6ynM6cKYj06A"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 2️⃣ List available models
print("Available models for generateContent:")
models = genai.list_models()
for m in models:
    if 'generateContent' in getattr(m, 'supported_generation_methods', []):
        print(f"- {m.name} ({m.display_name})")

# 3️⃣ Pick a supported model (from the list above)
model_name = "models/gemini-2.5-pro-preview-03-25"
model = genai.GenerativeModel(model_name)

# 4️⃣ Generate content
prompt = "Say hello from Gemini!"
response = model.generate_content(prompt)

# 5️⃣ Print the generated content
print("\n=== Generated Content ===")
print(response.result[0].content[0].text)
