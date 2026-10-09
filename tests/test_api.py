
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Energy Consumption Prediction API is running"
    assert data["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_rejects_missing_input():
    response = client.post("/predict", json={})

    assert response.status_code == 422



def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "lights": 0,
            "T1": 24.2,
            "RH_1": 39.79,
            "T2": 22.5,
            "RH_2": 41.09,
            "T3": 25.76,
            "RH_3": 39.0,
            "T4": 24.1,
            "RH_4": 38.1633333333,
            "T5": 22.29,
            "RH_5": 49.79,
            "T6": 14.04,
            "RH_6": 29.874,
            "T7": 23.83,
            "RH_7": 41.536,
            "T8": 24.5,
            "RH_8": 44.6633333333,
            "T9": 22.6,
            "RH_9": 45.678,
            "T_out": 13.7333333333,
            "Press_mm_hg": 751.3833333333,
            "RH_out": 81.1666666667,
            "Windspeed": 1,
            "Visibility": 28.3333333333,
            "Tdewpoint": 10.5333333333,
            "rv1": 20.1062472304,
            "rv2": 20.1062472304,
            "hour": 5,
            "day_of_week": 6,
            "month": 5,
            "is_weekend": 1,
            "hour_sin": 0.9659258263,
            "hour_cos": 0.2588190451,
            "day_sin": -0.7818314825,
            "day_cos": 0.6234898019,
            "lag_1h": 60,
            "lag_24h": 50,
            "lag_7d": 50,
            "rolling_mean_1h": 56.6666666667,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_appliances_consumption" in data
    assert data["predicted_appliances_consumption"] >= 0
    assert data["unit"] == "Wh"
    assert data["model"] == "Linear Regression"     