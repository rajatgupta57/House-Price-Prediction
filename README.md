# House Price Prediction Using Machine Learning

A simple, end-to-end Machine Learning project that predicts the price of a
residential property from its location, area, number of bedrooms (BHK), and
number of bathrooms. Built as a B.Tech Summer Training project applying
concepts studied during the **Amazon ML Summer School 2026**.

> This is a **general house price prediction system**. The dataset used
> happens to describe properties in Delhi, but the project itself is not
> limited to Delhi — the same code works with any similarly structured
> housing dataset.

---

## 1. Project Description

Manually estimating a property's price is difficult because it depends on
several factors at once (location, size, number of rooms, etc.). This
project builds a supervised Machine Learning regression pipeline that learns
the relationship between a small set of property characteristics and price,
and exposes the trained model through a simple Streamlit web app.

## 2. Dataset Information

- **Source:** https://github.com/tarunsingamsetti52/House_price_prediction
- **File:** `Delhi house data.csv`
- **Raw columns in the file:** `Area, BHK, Bathroom, Furnishing, Locality, Parking, Price, Status, Transaction, Type, Per_Sqft`
- **Columns actually used by this project:**
  - `Area` — property area in square feet (numeric input)
  - `BHK` — number of bedrooms (numeric input)
  - `Bathroom` — number of bathrooms (numeric input)
  - `Locality` — property location (categorical input)
  - `Price` — target variable
- The remaining columns (`Furnishing`, `Parking`, `Status`, `Transaction`,
  `Type`, `Per_Sqft`) exist in the raw file but are **not** used as model
  inputs, to keep the project scope simple as intended.
- **Price unit:** the source dataset does not document a unit label on the
  `Price` column; treat the values as being in the same currency/unit as the
  original CSV (commonly Indian Rupees for this dataset) and verify this
  against the raw file before quoting a unit in your report.

## 3. Features

| Role   | Feature    | Type        |
|--------|-----------|-------------|
| Input  | Locality   | Categorical |
| Input  | Area       | Numerical   |
| Input  | BHK        | Numerical   |
| Input  | Bathroom   | Numerical   |
| Target | Price      | Numerical   |

## 4. Machine Learning Approach

- **Type:** Supervised Learning — Regression
- **Algorithm:** Linear Regression (the only model used, by design)
- **Preprocessing:** scikit-learn `ColumnTransformer` inside a `Pipeline`
  - Numeric features (`Area`, `BHK`, `Bathroom`): median imputation + standard scaling
  - Categorical feature (`Locality`): most-frequent imputation + One-Hot Encoding
- **Split:** 80% training / 20% testing, `random_state=42` for reproducibility
- **Evaluation metrics:** R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

Preprocessing is fitted only on the training data (inside the pipeline) and
then applied to the test data and to new user input, so there is no data
leakage between training and evaluation.

## 5. Technologies Used

| Tool/Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Matplotlib | EDA and result visualisation |
| Scikit-learn | Preprocessing, Linear Regression, evaluation metrics |
| Streamlit | Web application interface |
| Joblib | Saving/loading the trained pipeline |
| Jupyter Notebook | Exploratory analysis / step-by-step walkthrough |

## 6. Project Structure

```text
house-price-prediction/
│
├── data/
│   ├── README.md              <- instructions for placing the dataset
│   └── Delhi house data.csv   <- (you add this; not committed to git)
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── model/                     <- created by train_model.py
│   ├── house_price_model.pkl
│   ├── localities.json
│   └── results.json
│
├── images/                    <- created by train_model.py
│   ├── price_distribution.png
│   ├── area_vs_price.png
│   ├── bhk_vs_price.png
│   ├── bathrooms_vs_price.png
│   ├── location_vs_price.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── src/
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Installation

```bash
# 1. Clone or download this project, then move into it
cd house-price-prediction

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 8. How to Train the Model

1. Download `Delhi house data.csv` from
   https://github.com/tarunsingamsetti52/House_price_prediction and place it
   inside the `data/` folder (see `data/README.md`).
2. Run:

   ```bash
   python src/train_model.py
   ```

3. The script will print dataset inspection details, cleaning steps, and the
   final evaluation metrics to the terminal, and will save:
   - the trained pipeline to `model/house_price_model.pkl`
   - the list of known localities to `model/localities.json`
   - the evaluation results to `model/results.json`
   - all EDA / evaluation plots to `images/`

## 9. How to Run the Streamlit App

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) —
open it in your browser.

## 10. How to Deploy (Streamlit Community Cloud)

See the **Deployment** section below for the exact steps.

## 11. Example Application Workflow

1. User opens the Streamlit app.
2. User selects a **Location** from the dropdown, and enters **Area (sqft)**,
   **BHK**, and **Number of Bathrooms**.
3. User clicks **Predict Price**.
4. The app validates the input, passes it through the same preprocessing
   pipeline used during training, and displays:
   `Estimated House Price: ₹...`

## 12. Model Evaluation Metrics

**These are the actual, verified results obtained by running `src/train_model.py`
on the real `Delhi house data.csv` dataset (1259 raw rows).**

| Metric | Result |
|---|---:|
| Original Records | 1259 |
| Records After Cleaning | 1139 |
| Training Samples | 911 |
| Testing Samples | 228 |
| Train/Test Split | 80:20 |
| Input Features (raw) | 4 (Area, BHK, Bathroom, Locality) |
| Features After One-Hot Encoding | 320 |
| Unique Localities | 365 |
| **R² Score** | **0.5790** |
| **MAE** | **7,701,833.70** |
| **RMSE** | **16,172,162.21** |

**Honest interpretation:** an R² of ~0.58 means the model explains a
moderate — not excellent — share of the price variation. Two real,
verifiable reasons are visible in `images/`:

1. **Very high location cardinality relative to data size** — 365 distinct
   `Locality` values across only 1139 rows means many localities appear
   only once or twice, so One-Hot Encoding turns 1 categorical column into
   320 mostly-sparse columns, which a plain Linear Regression cannot use
   very efficiently.
2. **A long right tail of expensive outlier properties** — see
   `images/price_distribution.png` and `images/actual_vs_predicted.png`;
   a handful of properties priced far above the rest pull the RMSE up
   sharply, since RMSE penalises large errors more heavily than MAE.

This is reported as obtained, without adjustment, in line with the
project's own rule not to invent or hide results.

---

## Deployment (Streamlit Community Cloud)

1. Create a new GitHub repository (e.g. `house-price-prediction`).
2. Push this entire project to it, **including** the real
   `data/Delhi house data.csv` file (or adjust `.gitignore` if you'd rather
   keep the dataset private and load it another way) and the generated
   `model/house_price_model.pkl`, `model/localities.json`, and
   `model/results.json` files — the deployed app needs these to be present
   in the repo since Streamlit Community Cloud does not run your training
   script for you.
3. Make sure `requirements.txt` is present at the root of the repo.
4. Go to https://share.streamlit.io and sign in with GitHub.
5. Click **New app**, select your repository and branch.
6. Set the **Main file path** to `app.py`.
7. Click **Deploy**.
8. Streamlit Community Cloud will build the environment and launch the app,
   giving you a public URL such as `https://<your-app-name>.streamlit.app`.

(No deployment URL is provided here since deployment has not actually been
performed from this environment — you will get your own real URL once you
complete the steps above.)
