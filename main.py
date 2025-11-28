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
    """Modern HTML UI for house price prediction"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>House Price Predictor AI</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .header p {
                opacity: 0.9;
                font-size: 1.1rem;
            }
            
            .form-container {
                padding: 40px;
                background: white;
            }
            
            .form-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 30px;
                margin-bottom: 30px;
            }
            
            .form-group {
                position: relative;
            }
            
            .form-group label {
                display: block;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 0.95rem;
            }
            
            .form-group .icon {
                position: absolute;
                right: 15px;
                top: 50%;
                transform: translateY(-50%);
                color: #3498db;
                font-size: 1.2rem;
            }
            
            .form-group input {
                width: 100%;
                padding: 15px 50px 15px 15px;
                border: 2px solid #e1e8ed;
                border-radius: 12px;
                font-size: 1rem;
                transition: all 0.3s ease;
                background: #f8f9fa;
            }
            
            .form-group input:focus {
                outline: none;
                border-color: #3498db;
                background: white;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            }
            
            .predict-btn {
                width: 100%;
                padding: 18px;
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 1.2rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
                box-shadow: 0 8px 15px rgba(52, 152, 219, 0.3);
            }
            
            .predict-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 25px rgba(52, 152, 219, 0.4);
                background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
            }
            
            .predict-btn:active {
                transform: translateY(0);
            }
            
            .predict-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .result {
                margin-top: 30px;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                display: none;
                animation: slideInUp 0.5s ease-out;
            }
            
            .result.success {
                background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                color: white;
                box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
            }
            
            .result.error {
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                box-shadow: 0 8px 25px rgba(231, 76, 60, 0.3);
            }
            
            .result h3 {
                margin-bottom: 15px;
                font-size: 1.5rem;
            }
            
            .price {
                font-size: 3rem;
                font-weight: 700;
                margin: 15px 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            .details {
                opacity: 0.9;
                font-size: 1rem;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #e1e8ed;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .examples {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 12px;
                border-left: 4px solid #3498db;
            }
            
            .examples h4 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.1rem;
            }
            
            .example-btn {
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px 15px;
                margin: 5px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: all 0.3s ease;
            }
            
            .example-btn:hover {
                background: #3498db;
                color: white;
                border-color: #3498db;
            }
            
            @media (max-width: 768px) {
                .container { margin: 10px; }
                .form-container { padding: 20px; }
                .form-grid { grid-template-columns: 1fr; gap: 20px; }
                .header h1 { font-size: 2rem; }
                .price { font-size: 2.5rem; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-home"></i> AI House Price Predictor</h1>
                <p>Get instant property valuations using advanced machine learning</p>
            </div>
            
            <div class="form-container">
                <form id="predictionForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label><i class="fas fa-dollar-sign"></i> Median Income (tens of thousands)</label>
                            <input type="number" step="0.0001" value="8.3252" name="medInc" required>
                            <i class="fas fa-money-bill-wave icon"></i>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-calendar-alt"></i> House Age (years)</label>
                            <input type="number" step="0.01" value="41.0" name="houseAge" required>
                            <i class="fas fa-clock icon"></i>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-bed"></i> Average Rooms per House</label>
                            <input type="number" step="0.01" value="6.98" name="aveRooms" required>
                            <i class="fas fa-door-open icon"></i>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-bed"></i> Average Bedrooms per House</label>
                            <input type="number" step="0.01" value="1.02" name="aveBedrms" required>
                            <i class="fas fa-bed icon"></i>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-users"></i> Population in Area</label>
                            <input type="number" step="1" value="322" name="population" required>
                            <i class="fas fa-chart-line icon"></i>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-home"></i> Average Household Size</label>
                            <input type="number" step="0.01" value="2.55" name="aveOccup" required>
                            <i class="fas fa-family icon"></i>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-map-marker-alt"></i> Latitude</label>
                            <input type="number" step="0.01" value="37.88" name="latitude" required>
                            <i class="fas fa-globe icon"></i>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-map-marker-alt"></i> Longitude</label>
                            <input type="number" step="0.01" value="-122.23" name="longitude" required>
                            <i class="fas fa-compass icon"></i>
                        </div>
                    </div>
                    
                    <button type="submit" class="predict-btn">
                        <i class="fas fa-magic"></i> Predict House Price
                    </button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing property data...</p>
                </div>
                
                <div id="result" class="result">
                    <h3 id="resultTitle">Estimated Property Value</h3>
                    <div class="price" id="predictionPrice"></div>
                    <div class="details" id="predictionDetails"></div>
                </div>
                
                <div class="examples">
                    <h4><i class="fas fa-lightbulb"></i> Quick Examples</h4>
                    <button type="button" class="example-btn" onclick="loadExample([8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23])">
                        Bay Area Property
                    </button>
                    <button type="button" class="example-btn" onclick="loadExample([8.3014, 21.0, 6.24, 0.97, 2401, 2.11, 37.86, -122.22])">
                        Urban Location
                    </button>
                    <button type="button" class="example-btn" onclick="loadExample([7.2574, 52.0, 8.29, 1.07, 496, 2.80, 37.85, -122.24])">
                        Suburban Home
                    </button>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById('predictionForm');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const resultTitle = document.getElementById('resultTitle');
            const predictionPrice = document.getElementById('predictionPrice');
            const predictionDetails = document.getElementById('predictionDetails');
            const submitBtn = form.querySelector('.predict-btn');

            function loadExample(values) {
                const inputs = form.querySelectorAll('input');
                values.forEach((value, index) => {
                    inputs[index].value = value;
                });
            }

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                // Show loading
                loading.style.display = 'block';
                result.style.display = 'none';
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                
                const formData = new FormData(form);
                const data = [
                    parseFloat(formData.get('medInc')),
                    parseFloat(formData.get('houseAge')),
                    parseFloat(formData.get('aveRooms')),
                    parseFloat(formData.get('aveBedrms')),
                    parseFloat(formData.get('population')),
                    parseFloat(formData.get('aveOccup')),
                    parseFloat(formData.get('latitude')),
                    parseFloat(formData.get('longitude'))
                ];

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ data })
                    });
                    
                    const resultData = await response.json();
                    
                    // Hide loading
                    loading.style.display = 'none';
                    
                    if (resultData.error) {
                        result.className = 'result error';
                        resultTitle.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Prediction Error';
                        predictionPrice.textContent = 'Error';
                        predictionDetails.textContent = resultData.error;
                    } else {
                        result.className = 'result success';
                        resultTitle.innerHTML = '<i class="fas fa-chart-line"></i> Estimated Property Value';
                        predictionPrice.textContent = resultData.prediction_formatted;
                        predictionDetails.innerHTML = `
                            <strong>Raw Value:</strong> $${resultData.prediction.toLocaleString()}<br>
                            <strong>Confidence:</strong> Based on California housing market data
                        `;
                    }
                    
                    result.style.display = 'block';
                } catch (error) {
                    loading.style.display = 'none';
                    result.className = 'result error';
                    resultTitle.innerHTML = '<i class="fas fa-wifi"></i> Connection Error';
                    predictionPrice.textContent = 'Error';
                    predictionDetails.textContent = 'Unable to connect to prediction service';
                    result.style.display = 'block';
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-magic"></i> Predict House Price';
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