# Air Quality Index Prediction (Flask)

Simple Flask app that predicts AQI (`PM 2.5`) from weather inputs using an `ExtraTreesRegressor` model.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run the web app

```bash
python app.py
```

Open the app in your browser at `http://127.0.0.1:5000/`.

## 3) Retrain the model manually

Use the helper script to regenerate a model compatible with your current `scikit-learn` version:

```bash
python retrain_model.py
```

Optional arguments:

```bash
python retrain_model.py --data-path Dataset/Airquality_index.csv --model-path tree_gridcv.pkl --n-estimators 600 --random-state 100
```

## Notes

- The app loads `tree_gridcv.pkl` at startup.
- If loading fails due to old pickle/version incompatibility, `app.py` automatically retrains from `Dataset/Airquality_index.csv` and saves a fresh model.


## Troubleshooting

- **Model pickle version/dtype error** (`InconsistentVersionWarning` or incompatible node dtype):
  - Run `python retrain_model.py` to regenerate `tree_gridcv.pkl` with your current `scikit-learn`.
  - Then restart the app with `python app.py`.

- **Missing package errors** (`ModuleNotFoundError`):
  - Run `pip install -r requirements.txt`.
  - If needed, verify interpreter is the same one used by VS Code.

- **Dataset not found** while retraining:
  - Ensure `Dataset/Airquality_index.csv` exists.
  - Or pass a custom path: `python retrain_model.py --data-path <your_csv_path>`.

- **Prediction page shows server error**:
  - Ensure all form fields are numeric (`T`, `TM`, `Tm`, `SLP`, `H`, `VV`, `V`, `VM`).
  - Do not leave fields blank.
