from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

# 1. Load Environment & Config
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "house_price_model.pkl")
MODEL_PATH = os.path.join("models", MODEL_NAME)

# 2. Initialize API
app = FastAPI(
    title="PropVal AI API",
    description="Professional House Price Prediction API using Random Forest",
    version="1.0.0",
)

# 3. Load Model (Global variable for caching)
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# 4. Define Input Schema (Data Validation)
# This acts as a contract. If the user sends bad data, API rejects it.
class HousingFeatures(BaseModel):
    MedInc: float  # Median Income
    HouseAge: float  # Median House Age
    AveRooms: float  # Average Rooms
    AveBedrms: float  # Average Bedrooms
    Population: float  # Population
    AveOccup: float  # Average Occupancy
    Latitude: float  # Latitude
    Longitude: float  # Longitude

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 3.5,
                "HouseAge": 30.0,
                "AveRooms": 5.0,
                "AveBedrms": 1.1,
                "Population": 1500.0,
                "AveOccup": 3.0,
                "Latitude": 34.0,
                "Longitude": -118.0,
            }
        }


# 5. Define Endpoints


@app.get("/")
def home():
    """Redirects to Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    """Kubernetes-style health check."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Predicts house price based on input features.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model is not available")

    try:
        # Convert input Pydantic object to DataFrame
        input_data = pd.DataFrame([features.model_dump()])

        # Make prediction
        prediction = model.predict(input_data)

        # Return result (in $100k units as per dataset)
        return {"predicted_price": float(prediction[0]), "currency_unit": "100k USD"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
