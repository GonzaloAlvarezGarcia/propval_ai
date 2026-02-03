import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
MODEL_NAME = os.getenv("MODEL_NAME", "house_price_model.pkl")
MODEL_PATH = os.path.join("models", MODEL_NAME)
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 10))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))


def load_data():
    """Loads the California Housing dataset."""
    print("Loading data...")
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["PRICE"] = data.target
    return df


def train_model():
    """Trains a Random Forest Regressor using env config and saves it."""
    # Load Data
    df = load_data()
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train with Config
    print(f"Training model with n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Trained. MSE: {mse:.2f}, R2: {r2:.2f}")

    # Save Model
    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
