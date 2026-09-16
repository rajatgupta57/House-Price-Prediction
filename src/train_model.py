"""
train_model.py
================
House Price Prediction Using Machine Learning
Amazon ML Summer School 2026 - Summer Training Project

This script:
    1. Loads the housing dataset
    2. Inspects and cleans it
    3. Builds a preprocessing + Linear Regression pipeline
    4. Splits data into train/test (80:20)
    5. Trains the model
    6. Evaluates it on the test set (R2, MAE, RMSE)
    7. Saves plots (EDA + actual vs predicted + residuals) to images/
    8. Saves the trained pipeline to model/house_price_model.pkl
    9. Prints a clean results table you can paste into your report

Run from the project root:
    python src/train_model.py

NOTE ON DATASET COLUMNS
------------------------
The real "Delhi house data.csv" file (from the referenced GitHub repo /
the original Kaggle "Housing Price Dataset of Delhi (India)") uses these
column names:

    Area, BHK, Bathroom, Furnishing, Locality, Parking,
    Price, Status, Transaction, Type, Per_Sqft

This script only uses the four input columns the project actually needs
(Area, BHK, Bathroom, Locality) plus the target (Price). The other columns
(Furnishing, Parking, Status, Transaction, Type, Per_Sqft) are present in
the raw file but are NOT used as model inputs, exactly as scoped in the
project brief. The script is defensive about column names (see
`resolve_columns`) so it also works if your CSV copy uses the friendlier
names Area Name / Area in sqft / Number of Bathrooms.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so the script works regardless of
# whether it's launched from the root or from inside src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Delhi house data.csv"
MODEL_DIR = PROJECT_ROOT / "model"
IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_PATH = PROJECT_ROOT / "model" / "results.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print("=" * 70)
        print("ERROR: Dataset file not found.")
        print(f"Expected file at: {path}")
        print()
        print("Download 'Delhi house data.csv' from:")
        print("  https://github.com/tarunsingamsetti52/House_price_prediction")
        print("and place it inside the data/ folder, then re-run this script.")
        print("=" * 70)
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[1/9] Loaded dataset: {path.name}  -> shape = {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Resolve column names (handles both the real dataset's column
# names and the friendlier names some course briefs use)
# ---------------------------------------------------------------------------
def resolve_columns(df: pd.DataFrame) -> dict:
    """Return a mapping of logical role -> actual column name in df."""
    candidates = {
        "area": ["Area", "Area in sqft", "area", "Area_sqft"],
        "bhk": ["BHK", "bhk", "Bedrooms"],
        "bathroom": ["Bathroom", "Number of Bathrooms", "Bathrooms", "bathroom"],
        "location": ["Locality", "Area Name", "Location", "locality"],
        "price": ["Price", "price"],
    }
    resolved = {}
    for role, options in candidates.items():
        found = next((c for c in options if c in df.columns), None)
        if found is None:
            raise KeyError(
                f"Could not find a column for '{role}'. "
                f"Looked for: {options}. Columns present: {list(df.columns)}"
            )
        resolved[role] = found
    print(f"[2/9] Resolved columns: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Step 3: Inspect + clean
# ---------------------------------------------------------------------------
def inspect_and_clean(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    n_original = len(df)
    print(f"[3/9] Original record count: {n_original}")
    print(f"      Columns and dtypes:\n{df.dtypes}")
    print(f"      Missing values per column:\n{df.isnull().sum()}")
    print(f"      Duplicate rows: {df.duplicated().sum()}")

    work = df.copy()

    # Keep only the columns this project actually needs
    keep_cols = [cols["area"], cols["bhk"], cols["bathroom"], cols["location"], cols["price"]]
    work = work[keep_cols].copy()
    work.columns = ["Area", "BHK", "Bathroom", "Locality", "Price"]

    # Coerce numeric columns (the raw dataset sometimes stores these as
    # text with commas / currency-like formatting; this makes cleaning
    # robust without assuming the file is already perfectly numeric).
    for c in ["Area", "BHK", "Bathroom", "Price"]:
        work[c] = (
            work[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)")[0]
        )
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work["Locality"] = work["Locality"].astype(str).str.strip()

    # Drop duplicate rows
    before = len(work)
    work = work.drop_duplicates()
    print(f"      Removed {before - len(work)} duplicate row(s)")

    # Drop rows with a missing target - a row with no price cannot be used
    # for supervised training or evaluation.
    before = len(work)
    work = work.dropna(subset=["Price"])
    print(f"      Removed {before - len(work)} row(s) with missing Price")

    # Remove rows with missing values in the input features rather than
    # guessing values for them (a small, simple, explainable rule).
    before = len(work)
    work = work.dropna(subset=["Area", "BHK", "Bathroom", "Locality"])
    print(f"      Removed {before - len(work)} row(s) with missing input feature(s)")

    # Remove obviously invalid values: zero/negative area, BHK, bathrooms,
    # or price. These cannot correspond to a real property and would only
    # add noise / distort the learned relationship.
    before = len(work)
    work = work[(work["Area"] > 0) & (work["BHK"] > 0) & (work["Bathroom"] > 0) & (work["Price"] > 0)]
    print(f"      Removed {before - len(work)} row(s) with zero/negative Area, BHK, Bathroom, or Price")

    work = work.reset_index(drop=True)
    n_final = len(work)
    print(f"      Final record count after cleaning: {n_final} "
          f"({n_original - n_final} row(s) removed in total)")

    return work


# ---------------------------------------------------------------------------
# Step 4: Exploratory Data Analysis (saves plots to images/)
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, images_dir: Path):
    print("[4/9] Generating EDA plots...")

    # 1. Price distribution
    plt.figure(figsize=(7, 5))
    plt.hist(df["Price"], bins=40, color="#3B6EA5", edgecolor="white")
    plt.title("Distribution of House Prices")
    plt.xlabel("Price")
    plt.ylabel("Number of Properties")
    plt.tight_layout()
    plt.savefig(images_dir / "price_distribution.png", dpi=150)
    plt.close()

    # 2. Area vs Price
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Area"], df["Price"], alpha=0.4, s=15, color="#C87A25")
    plt.title("Area (sqft) vs Price")
    plt.xlabel("Area in sqft")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(images_dir / "area_vs_price.png", dpi=150)
    plt.close()

    # 3. BHK vs Price
    plt.figure(figsize=(7, 5))
    df.boxplot(column="Price", by="BHK", ax=plt.gca())
    plt.title("BHK vs Price")
    plt.suptitle("")
    plt.xlabel("BHK")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(images_dir / "bhk_vs_price.png", dpi=150)
    plt.close()

    # 4. Bathrooms vs Price
    plt.figure(figsize=(7, 5))
    df.boxplot(column="Price", by="Bathroom", ax=plt.gca())
    plt.title("Number of Bathrooms vs Price")
    plt.suptitle("")
    plt.xlabel("Number of Bathrooms")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(images_dir / "bathrooms_vs_price.png", dpi=150)
    plt.close()

    # 5. Location-wise price (top 10 localities by frequency, only if
    # there are more than a handful of distinct localities)
    top_localities = df["Locality"].value_counts().head(10).index
    subset = df[df["Locality"].isin(top_localities)]
    if len(top_localities) >= 2:
        plt.figure(figsize=(9, 5))
        order = subset.groupby("Locality")["Price"].median().sort_values(ascending=False).index
        subset.boxplot(column="Price", by="Locality", ax=plt.gca(), rot=45)
        plt.title("Price by Locality (Top 10 most frequent localities)")
        plt.suptitle("")
        plt.xlabel("Locality")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.savefig(images_dir / "location_vs_price.png", dpi=150)
        plt.close()

    print(f"      Saved plots to: {images_dir}")


# ---------------------------------------------------------------------------
# Step 5: Build preprocessing + model pipeline
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    numeric_features = ["Area", "BHK", "Bathroom"]
    categorical_features = ["Locality"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Step 6: Train + evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df[["Area", "BHK", "Bathroom", "Locality"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[5/9] Train/Test split -> train={len(X_train)}  test={len(X_test)}  "
          f"({int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)})")

    pipeline = build_pipeline()
    print("[6/9] Training Linear Regression model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    n_features_after_encoding = pipeline.named_steps["preprocessor"].transform(X_train).shape[1]

    print("[7/9] Evaluation on held-out test set:")
    print(f"      R2 Score : {r2:.4f}")
    print(f"      MAE      : {mae:,.2f}")
    print(f"      RMSE     : {rmse:,.2f}")

    results = {
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "train_test_split": f"{int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}",
        "num_input_features_raw": 4,
        "num_features_after_encoding": int(n_features_after_encoding),
        "num_categorical_features": 1,
        "num_numerical_features": 3,
        "r2_score": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "num_unique_localities": int(df["Locality"].nunique()),
    }
    return pipeline, X_test, y_test, y_pred, results


# ---------------------------------------------------------------------------
# Step 7: Actual vs Predicted + Residual plots
# ---------------------------------------------------------------------------
def plot_predictions(y_test, y_pred, images_dir: Path):
    print("[8/9] Generating Actual vs Predicted and residual plots...")

    # Actual vs Predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.4, s=15, color="#3B6EA5")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, color="#C0392B", linewidth=1.5, label="Perfect Prediction Line")
    plt.title("Actual Price vs Predicted Price")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.4, s=15, color="#4F8F55")
    plt.axhline(0, color="#C0392B", linewidth=1.5)
    plt.title("Residual Plot (Actual - Predicted)")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(images_dir / "residual_plot.png", dpi=150)
    plt.close()

    print(f"      Saved: actual_vs_predicted.png, residual_plot.png -> {images_dir}")


# ---------------------------------------------------------------------------
# Step 8: Save pipeline + results
# ---------------------------------------------------------------------------
def save_artifacts(pipeline: Pipeline, df: pd.DataFrame, results: dict):
    model_path = MODEL_DIR / "house_price_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"[9/9] Saved trained pipeline to: {model_path}")

    # Save the list of known localities separately too, so the Streamlit
    # app can populate its dropdown without having to load the full
    # dataset at runtime.
    localities = sorted(df["Locality"].unique().tolist())
    with open(MODEL_DIR / "localities.json", "w") as f:
        json.dump(localities, f, indent=2)
    print(f"      Saved {len(localities)} known localities to: model/localities.json")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"      Saved evaluation results to: {RESULTS_PATH}")


def print_report_table(results: dict, dataset_name: str, n_original: int, n_final: int):
    print("\n" + "=" * 60)
    print("REPORT RESULTS TABLE (copy into your training report)")
    print("=" * 60)
    rows = [
        ("Dataset File", dataset_name),
        ("Original Records", n_original),
        ("Records After Cleaning", n_final),
        ("Training Samples", results["training_samples"]),
        ("Testing Samples", results["testing_samples"]),
        ("Train/Test Split", results["train_test_split"]),
        ("Input Features (raw)", results["num_input_features_raw"]),
        ("Features After Encoding", results["num_features_after_encoding"]),
        ("Numerical Features", results["num_numerical_features"]),
        ("Categorical Features", results["num_categorical_features"]),
        ("Unique Localities", results["num_unique_localities"]),
        ("R2 Score", f"{results['r2_score']:.4f}"),
        ("MAE", f"{results['mae']:.2f}"),
        ("RMSE", f"{results['rmse']:.2f}"),
    ]
    width1 = max(len(r[0]) for r in rows) + 2
    for label, value in rows:
        print(f"{label:<{width1}}: {value}")
    print("=" * 60 + "\n")


def main():
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - TRAINING SCRIPT")
    print("=" * 70)

    raw_df = load_dataset(DATA_PATH)
    n_original = len(raw_df)
    cols = resolve_columns(raw_df)
    clean_df = inspect_and_clean(raw_df, cols)
    n_final = len(clean_df)

    run_eda(clean_df, IMAGES_DIR)

    pipeline, X_test, y_test, y_pred, results = train_and_evaluate(clean_df)

    plot_predictions(y_test, y_pred, IMAGES_DIR)

    save_artifacts(pipeline, clean_df, results)

    print_report_table(results, DATA_PATH.name, n_original, n_final)

    print("Done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
