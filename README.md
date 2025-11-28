# 🏠 House Price Prediction API

A FastAPI web application with Gradio interface for predicting house prices using machine learning.

## Features

- **FastAPI Backend**: RESTful API endpoints for predictions
- **Gradio UI**: Interactive web interface at `/gradio`
- **Machine Learning**: Linear Regression model trained on California housing dataset
- **Cloud Ready**: Deployable on Render.com

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Predict house price
- `GET /gradio` - Interactive Gradio interface

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --reload
```

3. Visit:
- API docs: http://localhost:8000/docs
- Gradio UI: http://localhost:8000/gradio

## Model Features

The model uses 8 features from the California housing dataset:
1. Median Income
2. House Age
3. Average Rooms
4. Average Bedrooms
5. Population
6. Average Occupancy
7. Latitude
8. Longitude

## Deployment

This app is designed to be deployed on [Render.com](https://render.com) with the following settings:

- **Runtime**: Python 3
- **Build Command**: (leave blank)
- **Start Command**: `uvicorn main:app --host=0.0.0.0 --port=10000`

## Example Usage

### API Request
```bash
curl -X POST "https://your-app.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]}'
```

### Response
```json
{
  "prediction": 452500.0,
  "input_features": [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]
}
```