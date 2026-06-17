# ⚡ Joulebox

**Smart Home Energy Management System — Load Forecasting & Electricity Theft Detection using Machine Learning**

Joulebox is a machine learning project that analyzes smart home energy consumption data to do two things: predict how much energy a household will use (for smart grid load balancing), and flag unusual consumption patterns that could indicate electricity theft or equipment faults. Built as a Python for Data Science mini project, it combines a full ML pipeline with an interactive Streamlit dashboard.

🔗 **Live App:** [smart-home-energy-analytics.streamlit.app](https://smart-home-energy-analytics-jfnxjyklhugtv7bkz46nhx.streamlit.app/)

---

## What it does

Joulebox is built around two core tasks, both running on the [HomeC dataset](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information) — roughly 500,000 rows of 1-minute interval smart home energy readings paired with weather data.

**Task 1 — Smart Grid Load Balancing (Regression)**
Predicts household energy consumption (`use [kW]`) from environmental and time-based features like temperature, humidity, hour of day, month, and day of week. Four models are trained and compared: Linear Regression, Ridge Regression, Random Forest, and XGBoost — including an optional GridSearchCV step to auto-tune XGBoost's hyperparameters.

**Task 2 — Electricity Theft Detection (Anomaly Detection)**
Flags abnormal consumption readings using the IQR (Interquartile Range) method, surfacing records that fall well outside typical usage bounds — potential signs of meter tampering, theft, or equipment malfunction.

---

## Repo structure

```
.
├── .devcontainer/        # Dev container config for consistent environment setup
├── .gitignore
├── HomeC_sample.csv      # Smaller sample of the HomeC dataset, used by app.py
├── HomeC_Test.csv        # Smaller test split of the dataset
├── app.ipynb             # Jupyter notebook — full ML pipeline, EDA, model training
├── app.py                # Streamlit dashboard (multi-page app)
├── create_sample.py      # Script used to generate the sample/test CSVs from the full dataset
├── requirements.txt      # Python dependencies
└── README.md
```

---

## The dashboard (`app.py`)

A multi-page Streamlit app with:

- **Home** — project overview and dataset snapshot
- **EDA** — consumption distribution, correlation heatmap, hourly/monthly trends
- **Model Training** — pick an algorithm (Linear, Ridge, Random Forest, XGBoost), tune hyperparameters manually or via GridSearchCV, and see live MAE / RMSE / R² results
- **Anomaly Detection** — IQR-based theft/fault detection with visualizations and flagged records
- **Model Comparison** — side-by-side R² comparison across all four models
- **Live Prediction** — enter conditions (temperature, humidity, hour) and get an instant consumption prediction

---

## Tech stack

- **Language:** Python
- **ML:** scikit-learn (Linear Regression, Ridge, Random Forest), XGBoost
- **Data:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **App/Deployment:** Streamlit, Streamlit Community Cloud

---

## Running locally

```bash
git clone <repo-url>
cd joulebox
pip install -r requirements.txt
streamlit run app.py
```

Make sure `HomeC_sample.csv` is in the same directory as `app.py` — the app loads it directly by filename.

---

## Notes on the dataset

- Timestamps in the raw HomeC data are Unix seconds. They must be parsed with `pd.to_datetime(df['time'], unit='s')` — without `unit='s'`, pandas defaults to nanoseconds and silently misparses every row to 1970, corrupting the `hour`, `month`, and `dayofweek` features used by every model.
- For regression, outliers are IQR-capped to improve model accuracy. For anomaly detection, outliers are preserved — since abnormal readings are exactly what the theft-detection task is looking for.
- GridSearchCV (for XGBoost tuning) is run on a capped sample of the training data to keep search time reasonable on Streamlit Community Cloud's free-tier compute; the best parameters found are then refit on the full training set for final evaluation.

---

## Acknowledgements

Built for the Python for Data Science mini project, SY ECE (AIML), DES Pune University, under the guidance of Dr. Prajakta Khadakikar and Dr. Mosam Sangole.

Dataset: [HomeC — Smart Home Dataset with Weather Information](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information) (Kaggle).
