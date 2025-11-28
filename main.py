from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
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

class Input(BaseModel):
    data: Optional[List[float]] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.get("/")
def read_root():
    return {
        "message": "House Price Prediction API", 
        "status": "running",
        "model": "Simple Linear Regression",
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs",
            "ui": "/ui"
        }
    }

@app.post("/predict")
def predict(input: Input = Input()):
    try:
        prediction = predict_house_price_simple(input.data)
        if prediction is None:
            return {"error": "Prediction failed"}
        
        return {
            "prediction": float(prediction),
            "prediction_formatted": f"${prediction:,.2f}",
            "input_features": input.data,
            "feature_names": FEATURE_NAMES
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/ui")
def get_ui():
    """Simple HTML UI for testing the prediction"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Price Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input { width: 100%; padding: 8px; margin-bottom: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏠 House Price Predictor</h1>
            <p>Enter housing features to predict the median house value.</p>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label>Median Income:</label>
                    <input type="number" step="0.0001" value="8.3252" name="medInc">
                </div>
                <div class="form-group">
                    <label>House Age:</label>
                    <input type="number" step="0.01" value="41.0" name="houseAge">
                </div>
                <div class="form-group">
                    <label>Average Rooms:</label>
                    <input type="number" step="0.01" value="6.98" name="aveRooms">
                </div>
                <div class="form-group">
                    <label>Average Bedrooms:</label>
                    <input type="number" step="0.01" value="1.02" name="aveBedrms">
                </div>
                <div class="form-group">
                    <label>Population:</label>
                    <input type="number" step="1" value="322" name="population">
                </div>
                <div class="form-group">
                    <label>Average Occupancy:</label>
                    <input type="number" step="0.01" value="2.55" name="aveOccup">
                </div>
                <div class="form-group">
                    <label>Latitude:</label>
                    <input type="number" step="0.01" value="37.88" name="latitude">
                </div>
                <div class="form-group">
                    <label>Longitude:</label>
                    <input type="number" step="0.01" value="-122.23" name="longitude">
                </div>
                <button type="submit">Predict Price</button>
            </form>
            
            <div id="result" class="result" style="display:none;">
                <h3>Prediction Result:</h3>
                <p id="predictionText"></p>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const form = e.target;
                const data = [
                    parseFloat(form.medInc.value),
                    parseFloat(form.houseAge.value),
                    parseFloat(form.aveRooms.value),
                    parseFloat(form.aveBedrms.value),
                    parseFloat(form.population.value),
                    parseFloat(form.aveOccup.value),
                    parseFloat(form.latitude.value),
                    parseFloat(form.longitude.value)
                ];

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ data })
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        document.getElementById('predictionText').innerHTML = `Error: ${result.error}`;
                    } else {
                        document.getElementById('predictionText').innerHTML = 
                            `<strong>${result.prediction_formatted}</strong><br>
                             <small>Raw value: ${result.prediction}</small>`;
                    }
                    
                    document.getElementById('result').style.display = 'block';
                } catch (error) {
                    document.getElementById('predictionText').innerHTML = `Error: ${error.message}`;
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    # Run FastAPI server
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)