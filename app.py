from pathlib import Path
import warnings

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)

MODEL_PATH = Path("tree_gridcv.pkl")
DATA_PATH = Path("Dataset") / "Airquality_index.csv"
FEATURE_COLUMNS = ["T", "TM", "Tm", "SLP", "H", "VV", "V", "VM"]
TARGET_COLUMN = "PM 2.5"


def train_and_persist_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH.resolve()}")

    data = pd.read_csv(DATA_PATH).dropna(subset=[TARGET_COLUMN])
    model_instance = ExtraTreesRegressor(
        n_estimators=600,
        max_features="log2",
        random_state=100,
        n_jobs=-1,
    )
    model_instance.fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])

    joblib.dump(model_instance, MODEL_PATH, compress=3)

    return model_instance


def load_model():
    if MODEL_PATH.exists():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                return joblib.load(MODEL_PATH)
        except Exception as exc:
            print(
                "Model load failed due to compatibility issue. "
                f"Retraining with local dataset. Details: {exc}"
            )

    return train_and_persist_model()


model = load_model()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_frame = pd.DataFrame(
        [{column: float(request.form[column]) for column in FEATURE_COLUMNS}]
    )
    prediction = model.predict(input_frame)
    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run()
