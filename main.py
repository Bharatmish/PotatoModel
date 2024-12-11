from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow all origins for production. Update for stricter CORS as needed.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and define class names
MODEL = tf.keras.models.load_model("./saved_models/1")  # Path is relative to the api folder
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# OpenAI API Configuration
API_KEY = os.getenv("API_KEY")  # Retrieve API key from environment
ENDPOINT = os.getenv("API_ENDPOINT")  # Retrieve API endpoint from environment

if not API_KEY or not ENDPOINT:
    raise ValueError("API_KEY or API_ENDPOINT is not set in the environment.")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def generate_summary(disease_name, language):
    """
    Generate a detailed summary for the farmer in the selected language.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    prompt = (
        f"The detected disease in the potato plant leaf is {disease_name}. "
        "Generate a detailed summary for the farmer. Include: "
        "1. Disease causes and symptoms. "
        "2. Recommended insecticides and pesticides for treatment. "
        "3. Precautions and preventive measures. "
        "4. Any additional advice for the farmer."
        f"\nPlease provide the summary in {language}."
    )
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an agricultural expert helping farmers manage plant diseases."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        return summary
    except requests.RequestException as e:
        return f"Error generating summary: {e}"

@app.get("/ping")
async def ping():
    """
    Simple health-check endpoint.
    """
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Query("English")):
    """
    Accepts an image file and a language query parameter.
    Returns the predicted disease, confidence, and summary in the selected language.
    """
    # Read and preprocess the image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Make a prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    # Generate the summary for the predicted class in the selected language
    summary = generate_summary(predicted_class, language)
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'summary': summary,
        'language': language
    }

# Vercel requires the app to be accessible as `app` in the entry file.
