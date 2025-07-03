from fastapi import FastAPI, HTTPException
from fastapi import Request
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from contextlib import asynccontextmanager

# Define paths dynamically for Docker vs. local
import os

# Check if running inside Docker by presence of the mounted folder inside container
if os.path.exists("/app/models/logistic_regression.pkl"):
    model_path = "/app/models/logistic_regression.pkl"
    data_path = "/app/data/final_model_data.csv"
else:
    # Local machine paths (adjust as needed)
    model_path = r"C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\models\\logistic_regression.pkl"
    data_path = r"C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\data\\final_model_data.csv"

import joblib
import pandas as pd

model = joblib.load(model_path)
df = pd.read_csv(data_path)


df = pd.read_csv(data_path)

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")
    if not os.path.exists(data_path):
        raise RuntimeError(f"Data file not found at {data_path}")
    
    # Load model and data
    app.state.model = joblib.load(model_path)
    app.state.df = pd.read_csv(data_path)
    
    yield  # Startup complete
    # You can add shutdown cleanup here if needed

app = FastAPI(lifespan=lifespan)

# API Schemas
class CustomerIDRequest(BaseModel):
    customer_id: str

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float

@app.get("/")
def health_check():
    return {"status": "API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: CustomerIDRequest, req: Request):
    model = req.app.state.model
    df = req.app.state.df

    row = df[df["CustomerId"] == request.customer_id]

    if row.empty:
        raise HTTPException(status_code=404, detail="Customer ID not found")

    features = row.drop(columns=["CustomerId", "is_high_risk"])
    risk = model.predict_proba(features)[0][1]

    return PredictionResponse(customer_id=request.customer_id, risk_probability=risk)
