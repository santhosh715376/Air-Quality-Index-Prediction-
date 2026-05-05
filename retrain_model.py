import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

FEATURE_COLUMNS = ["T", "TM", "Tm", "SLP", "H", "VV", "V", "VM"]
TARGET_COLUMN = "PM 2.5"
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(raw_path: str) -> Path:
    parsed_path = Path(raw_path)
    if parsed_path.is_absolute():
        return parsed_path
    return (PROJECT_ROOT / parsed_path).resolve()


def train_and_save_model(
    data_path: Path,
    model_path: Path,
    n_estimators: int = 600,
    random_state: int = 100,
):
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    frame = pd.read_csv(data_path).dropna(subset=[TARGET_COLUMN])
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_features="log2",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    return model_path, len(frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain AQI ExtraTrees model and save it as a pickle file.")
    parser.add_argument(
        "--data-path",
        default="Dataset/Airquality_index.csv",
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--model-path",
        default="tree_gridcv.pkl",
        help="Path where the trained model pickle should be saved.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=600,
        help="Number of trees in ExtraTreesRegressor.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=100,
        help="Random state used for model training.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_path = resolve_project_path(args.data_path)
    model_path = resolve_project_path(args.model_path)

    saved_model_path, row_count = train_and_save_model(
        data_path=data_path,
        model_path=model_path,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print(f"Model trained and saved to: {saved_model_path}")
    print(f"Rows used for training: {row_count}")


if __name__ == "__main__":
    main()
