from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import joblib
import gradio as gr
import uvicorn
import os

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Load the trained model
try:
    model = joblib.load("house_model.pkl")
except FileNotFoundError:
    print("Warning: house_model.pkl not found. Please ensure the model file exists.")
    model = None

class Input(BaseModel):
    data: Optional[List[float]] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API", "status": "running"}

@app.post("/predict")
def predict(input: Input = Input()):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        pred = model.predict([input.data])
        return {"prediction": float(pred[0]), "input_features": input.data}
    except Exception as e:
        return {"error": str(e)}

# Gradio interface function
def predict_house_price(median_income, house_age, avg_rooms, avg_bedrooms, 
                       population, avg_occupancy, latitude, longitude):
    if model is None:
        return "Model not loaded"
    
    try:
        features = [median_income, house_age, avg_rooms, avg_bedrooms, 
                   population, avg_occupancy, latitude, longitude]
        prediction = model.predict([features])
        return f"${prediction[0]:,.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
def create_gradio_interface():
    interface = gr.Interface(
        fn=predict_house_price,
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
        title="🏠 House Price Predictor",
        description="Enter the housing features to predict the median house value.",
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