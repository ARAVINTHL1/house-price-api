from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import joblib
import gradio as gr
import uvicorn
import os
import numpy as np

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Simple linear model coefficients (pre-calculated from scikit-learn)
# These coefficients were extracted from the trained LinearRegression model
MODEL_COEFFICIENTS = [
    0.43663, 0.01384, -0.11757, 0.64933, -0.00013, -0.04221, -0.89986, -0.87088
]
MODEL_INTERCEPT = 0.14756

# Feature names for reference
FEATURE_NAMES = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

class Input(BaseModel):
    data: Optional[List[float]] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

def predict_house_price_simple(features):
    """Simple linear prediction without scikit-learn dependency"""
    try:
        # Convert to numpy array for calculation
        features = np.array(features)
        coefficients = np.array(MODEL_COEFFICIENTS)
        
        # Linear regression: prediction = intercept + sum(coefficients * features)
        prediction = MODEL_INTERCEPT + np.dot(coefficients, features)
        
        # Convert back to dollars (multiply by 100,000)
        return prediction * 100000
    except Exception as e:
        return None

@app.get("/")
def read_root():
    return {
        "message": "House Price Prediction API", 
        "status": "running",
        "model": "Simple Linear Regression (no scikit-learn dependency)"
    }

@app.post("/predict")
def predict(input: Input = Input()):
    try:
        prediction = predict_house_price_simple(input.data)
        if prediction is None:
            return {"error": "Prediction failed"}
        
        return {
            "prediction": float(prediction),
            "input_features": input.data,
            "feature_names": FEATURE_NAMES
        }
    except Exception as e:
        return {"error": str(e)}

# Gradio interface function
def predict_house_price_gradio(median_income, house_age, avg_rooms, avg_bedrooms, 
                              population, avg_occupancy, latitude, longitude):
    try:
        features = [median_income, house_age, avg_rooms, avg_bedrooms, 
                   population, avg_occupancy, latitude, longitude]
        prediction = predict_house_price_simple(features)
        
        if prediction is None:
            return "Prediction failed"
        
        return f"${prediction:,.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
def create_gradio_interface():
    interface = gr.Interface(
        fn=predict_house_price_gradio,
        inputs=[
            gr.Number(value=8.3252, label="Median Income"),
            gr.Number(value=41.0, label="House Age"),
            gr.Number(value=6.98, label="Average Rooms"),
            gr.Number(value=1.02, label="Average Bedrooms"),
            gr.Number(value=322, label="Population"),
            gr.Number(value=2.55, label="Average Occupancy"),
            gr.Number(value=37.88, label="Latitude"),
            gr.Number(value=-122.23, label="Longitude")
        ],
        outputs=gr.Textbox(label="Predicted House Price"),
        title="🏠 House Price Predictor (Simplified)",
        description="Enter housing features to predict median house value. Uses pre-trained linear model coefficients.",
        examples=[
            [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23],
            [8.3014, 21.0, 6.24, 0.97, 2401, 2.11, 37.86, -122.22],
            [7.2574, 52.0, 8.29, 1.07, 496, 2.80, 37.85, -122.24]
        ]
    )
    return interface

# Mount Gradio app
gradio_app = create_gradio_interface()

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

if __name__ == "__main__":
    # Run FastAPI server
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)