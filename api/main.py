

from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "baseline_model.joblib"

model = joblib.load(MODEL_PATH)
FEATURE_NAMES = list(model.feature_names_in_)

app = FastAPI(
    title="Energy Consumption Prediction API",
    description="Predict household appliance energy consumption.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    lights: float = Field(ge=0)
    T1: float
    RH_1: float
    T2: float
    RH_2: float
    T3: float
    RH_3: float
    T4: float
    RH_4: float
    T5: float
    RH_5: float
    T6: float
    RH_6: float
    T7: float
    RH_7: float
    T8: float
    RH_8: float
    T9: float
    RH_9: float
    T_out: float
    Press_mm_hg: float
    RH_out: float
    Windspeed: float
    Visibility: float
    Tdewpoint: float
    rv1: float
    rv2: float
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float
    lag_1h: float = Field(ge=0)
    lag_24h: float = Field(ge=0)
    lag_7d: float = Field(ge=0)
    rolling_mean_1h: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Energy Consumption Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    values = request.model_dump()

    input_df = pd.DataFrame(
        [[values[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    prediction = float(model.predict(input_df)[0])

    return {
        "predicted_appliances_consumption": round(
            max(0.0, prediction), 2
        ),
        "unit": "Wh",
        "model": "Linear Regression",
    }