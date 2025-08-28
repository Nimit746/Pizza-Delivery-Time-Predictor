from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Pizza Delivery Time Predictor",
    description="API for predicting pizza delivery times using machine learning",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://deliveryinsights.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the trained model
MODEL_PATH = "Model.pkl"

try:
    pizza_pred = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model file not found. Please make sure Model.pkl exists.")
    pizza_pred = None


# Pydantic models for request/response
class PizzaOrder(BaseModel):
    distance_miles: float = Field(..., gt=0, description="Distance in miles")
    pizza_count: int = Field(..., gt=0, description="Number of pizzas")
    day_of_week: str = Field(
        ...,
        description="Day of week (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)",
    )
    weather: str = Field(
        ..., description="Weather condition (Sunny, Cloudy, Rainy, Snowy)"
    )
    traffic_level: str = Field(..., description="Traffic level (Light, Medium, Heavy)")

    @validator("day_of_week")
    def validate_day(cls, v):
        # Convert to title case for case-insensitive matching
        v_title = v.strip().title()
        valid_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        if v_title not in valid_days:
            raise ValueError(
                f'day_of_week must be one of: {", ".join(valid_days)} (case insensitive)'
            )
        return v_title

    @validator("weather")
    def validate_weather(cls, v):
        # Convert to title case for case-insensitive matching
        v_title = v.strip().title()
        valid_weather = ["Sunny", "Cloudy", "Rainy", "Snowy"]
        if v_title not in valid_weather:
            raise ValueError(
                f'weather must be one of: {", ".join(valid_weather)} (case insensitive)'
            )
        return v_title

    @validator("traffic_level")
    def validate_traffic(cls, v):
        # Convert to title case for case-insensitive matching
        v_title = v.strip().title()
        valid_traffic = ["Light", "Medium", "Heavy"]
        if v_title not in valid_traffic:
            raise ValueError(
                f'traffic_level must be one of: {", ".join(valid_traffic)} (case insensitive)'
            )
        return v_title

    class Config:
        schema_extra = {
            "example": {
                "distance_miles": 2.5,
                "pizza_count": 2,
                "day_of_week": "Friday",
                "weather": "Sunny",
                "traffic_level": "Medium",
            }
        }


class PredictionResponse(BaseModel):
    predicted_time_minutes: float
    formatted_message: str
    order_details: dict


class BatchPredictionRequest(BaseModel):
    orders: List[PizzaOrder]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: dict


# Helper functions for converting string inputs to numeric values
def get_day_number(day_name: str) -> int:
    day_mapping = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7,
    }
    return day_mapping[day_name]


def get_weather_number(weather_name: str) -> int:
    weather_mapping = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Snowy": 4}
    return weather_mapping[weather_name]


def get_traffic_number(traffic_name: str) -> int:
    traffic_mapping = {"Light": 1, "Medium": 2, "Heavy": 3}
    return traffic_mapping[traffic_name]


def convert_order_to_numeric(order: PizzaOrder) -> dict:
    """Convert string-based order to numeric format for model prediction"""
    return {
        "distance_miles": order.distance_miles,
        "pizza_count": order.pizza_count,
        "day_of_week": get_day_number(order.day_of_week),
        "weather": get_weather_number(order.weather),
        "traffic_level": get_traffic_number(order.traffic_level),
    }


def format_prediction_response(
    order: PizzaOrder, predicted_time: float
) -> PredictionResponse:
    """Format the prediction response with all details"""
    order_details = {
        "distance_miles": order.distance_miles,
        "pizza_count": order.pizza_count,
        "day": order.day_of_week,
        "weather": order.weather,
        "traffic": order.traffic_level,
    }

    formatted_message = f"Your {order.pizza_count} pizza(s) will arrive in about {int(predicted_time)} minutes!"

    return PredictionResponse(
        predicted_time_minutes=round(predicted_time, 1),
        formatted_message=formatted_message,
        order_details=order_details,
    )


# API Endpoints
@app.get("/")
async def root():
    """Welcome message and API information"""
    return {
        "message": "🍕 Pizza Delivery Time Predictor API",
        "description": "Use /predict for single orders or /predict/batch for multiple orders",
        "model_status": "loaded" if pizza_pred is not None else "not_loaded",
        "endpoints": {
            "/predict": "POST - Predict delivery time for a single order",
            "/predict/batch": "POST - Predict delivery times for multiple orders",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pizza_pred is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_delivery_time(order: PizzaOrder):
    """Predict delivery time for a single pizza order"""

    if pizza_pred is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check if Model.pkl exists.",
        )

    try:
        # Convert string inputs to numeric format for the model
        numeric_order = convert_order_to_numeric(order)

        # Create DataFrame for prediction
        order_data = pd.DataFrame([numeric_order])

        # Make prediction
        prediction = pizza_pred.predict(order_data)[0]

        # Format and return response
        return format_prediction_response(order, prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_delivery_times(request: BatchPredictionRequest):
    """Predict delivery times for multiple pizza orders"""

    if pizza_pred is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check if Model.pkl exists.",
        )

    if len(request.orders) == 0:
        raise HTTPException(status_code=400, detail="No orders provided")

    try:
        # Convert all orders to numeric format
        numeric_orders = [convert_order_to_numeric(order) for order in request.orders]

        # Create DataFrame for all orders
        orders_data = pd.DataFrame(numeric_orders)

        # Make predictions
        predictions = pizza_pred.predict(orders_data)

        # Format responses
        prediction_responses = [
            format_prediction_response(order, pred)
            for order, pred in zip(request.orders, predictions)
        ]

        # Calculate summary statistics
        summary = {
            "total_orders": len(request.orders),
            "average_delivery_time": round(np.mean(predictions), 1),
            "min_delivery_time": round(np.min(predictions), 1),
            "max_delivery_time": round(np.max(predictions), 1),
            "total_pizzas": sum(order.pizza_count for order in request.orders),
        }

        return BatchPredictionResponse(
            predictions=prediction_responses, summary=summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if pizza_pred is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Get model coefficients if available
        feature_names = [
            "distance_miles",
            "pizza_count",
            "day_of_week",
            "weather",
            "traffic_level",
        ]
        coefficients = {}

        if hasattr(pizza_pred, "coef_"):
            coefficients = dict(zip(feature_names, pizza_pred.coef_.tolist()))

        intercept = pizza_pred.intercept_ if hasattr(pizza_pred, "intercept_") else None

        return {
            "model_type": type(pizza_pred).__name__,
            "features": feature_names,
            "coefficients": coefficients,
            "intercept": intercept,
            "model_loaded": True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting model info: {str(e)}"
        )


# Add some utility endpoints for reference
@app.get("/reference/days")
async def get_day_reference():
    """Get reference for day of week values"""
    return {
        "accepted_values": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        "description": "Use day names (case insensitive) - e.g., 'monday', 'TUESDAY', 'Wednesday' all work",
    }


@app.get("/reference/weather")
async def get_weather_reference():
    """Get reference for weather values"""
    return {
        "accepted_values": ["Sunny", "Cloudy", "Rainy", "Snowy"],
        "description": "Use weather condition names (case insensitive) - e.g., 'sunny', 'CLOUDY', 'Rainy' all work",
    }


@app.get("/reference/traffic")
async def get_traffic_reference():
    """Get reference for traffic level values"""
    return {
        "accepted_values": ["Light", "Medium", "Heavy"],
        "description": "Use traffic level names (case insensitive) - e.g., 'light', 'MEDIUM', 'Heavy' all work",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
