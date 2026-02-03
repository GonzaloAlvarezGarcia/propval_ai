from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_root_redirect():
    """Verifies that the root endpoint redirects to the documentation"""
    response = client.get("/")
    assert response.status_code == 200


def test_health_check():
    """Verifies that the health check endpoint responds"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_flow():
    """Verifies that the model returns a valid prediction"""
    payload = {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert isinstance(response.json()["predicted_price"], float)
