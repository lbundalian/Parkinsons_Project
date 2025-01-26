from fastapi import FastAPI, Query, HTTPException, Body
import pandas as pd
from typing import List, Dict, Optional
from enum import Enum
from functools import wraps
import joblib
from dataclasses import dataclass
import pandas as pd
import os

### Define a dataclass for input parameters of the endpoint
@dataclass
class ModelInput:
    MDVP_Fo: float
    MDVP_Fhi: float
    MDVP_Flo: float
    MDVP_Jitter_percent: float
    MDVP_Jitter_Abs: float
    MDVP_PPQ: float
    MDVP_Shim: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    Spread1: float
    Spread2: float
    D2: float
    PPE: float





app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "test.pkl")
model = joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Parkinson's Predictive Model API"}

@app.post("/sayhello")
def say_hello(name: str):
    
    
    response = ""
    try:

        response = f"Hello, how are you {name}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if not response:
        raise HTTPException(status_code=404, detail="Prob")

    response = { "greetings": response }

    return response

@app.post("/predict")
async def predict(features: ModelInput = Body(...)):
    try:
        input_df = pd.DataFrame([features.__dict__])
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

