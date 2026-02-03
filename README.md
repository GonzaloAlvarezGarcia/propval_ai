# PropVal AI: End-to-End House Price Prediction API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-RandomForest-orange)

> **A production-ready MLOps project that exposes a Random Forest Regressor via a REST API, fully containerized with Docker.**

## Project Overview

This project demonstrates the complete lifecycle of a Machine Learning model, from training to deployment. Unlike simple notebooks, **PropVal AI** is architected as a robust software product using **FastAPI** for the backend and **Pydantic** for strict data validation.

### Key Features

- **Training Pipeline:** Automated script (`train_pipeline.py`) to fetch data, train, and serialize the model.
- **Robust API:** FastAPI implementation with automatic Swagger UI documentation.
- **Data Validation:** Pydantic v2 schemas ensure input integrity before it reaches the model.
- **Production Ready:** Dockerized environment for consistent deployment across any cloud provider.
- **Configurable:** Externalized configuration via `.env` files.

## Tech Stack

- **Core:** Python 3.10
- **ML Framework:** Scikit-Learn (Random Forest)
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker
- **Data Handling:** Pandas, Numpy

## Quick Start

### Option A: Local Execution (Using Makefile)

1.  **Clone & Install**

    ```bash
    git clone https://github.com/gonzaloalvarezgarcia/propval_ai.git
    cd propval_ai
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    make install
    ```

2.  **Train the Model**

    ```bash
    make train
    # Output: Model saved to models/house_price_model.pkl
    ```

3.  **Run the API**
    ```bash
    make run
    ```
    Access the documentation at: `http://localhost:8000/docs`

### Option B: Docker Execution

1.  **Build the Container**

    ```bash
    make docker-build
    ```

2.  **Run Container**
    ```bash
    make docker-run
    ```

## Project Structure

```text
propval_ai/
├── Dockerfile           # Container definition
├── Makefile             # Automation commands
├── models/              # Serialized ML models (.pkl)
├── requirements.txt     # Dependencies
└── src/
    ├── app.py           # FastAPI application endpoint
    └── train_pipeline.py # ML Training script
```
