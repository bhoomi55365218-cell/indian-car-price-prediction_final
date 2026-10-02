import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Indian Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_pipeline.pkl")
TRAIN_PATH = os.path.join(DATA_DIR, "Cap_Training_Data_2025.csv")

FEATURES = [
    "Maker",
    "model",
    "Location",
    "Distance",
    "Owner Type",
    "manufacture_year",
    "Age of car",
    "engine_displacement",
    "engine_power",
    "body_type",
    "Vroom Audit Rating",
    "transmission",
    "door_count",
    "seat_count",
    "fuel_type",
]

CATEGORICAL = [
    "Maker",
    "model",
    "Location",
    "Owner Type",
    "body_type",
    "transmission",
    "fuel_type",
]

NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make common column-name variations match the training schema."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


@st.cache_data

def load_training_data():
    if not os.path.exists(TRAIN_PATH):
        return None
    return normalize_columns(pd.read_csv(TRAIN_PATH))


@st.cache_resource

def load_model():
    return joblib.load(MODEL_PATH)


train_df = load_training_data()

st.title("🚗 Indian Pre-Owned Car Price Predictor")
st.caption("Machine Learning capstone project — Streamlit deployment")

if train_df is None:
    st.error(
        "Training CSV not found. Put Cap_Training_Data_2025.csv inside the data/ folder."
    )
    st.stop()

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Save your trained preprocessing + model pipeline as "
        "models/car_price_pipeline.pkl"
    )
    st.info("The training notebook should create this file before you run the app.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

for col in NUMERICAL:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

for col in CATEGORICAL:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype("string")

single_tab, batch_tab, insights_tab = st.tabs(
    ["🔮 Single Prediction", "📁 Batch Prediction", "📊 Insights"]
)

with single_tab:
    st.subheader("Enter car details")

    col1, col2, col3 = st.columns(3)

    with col1:
        maker_options = sorted(train_df["Maker"].dropna().unique().tolist())
        maker = st.selectbox("Maker", maker_options)

        model_options = sorted(
            train_df.loc[train_df["Maker"] == maker, "model"].dropna().unique().tolist()
        )
        model_name = st.selectbox("Model", model_options)

        location_options = sorted(train_df["Location"].dropna().unique().tolist())
        location = st.selectbox("Location", location_options)

        owner_options = sorted(train_df["Owner Type"].dropna().unique().tolist())
        owner_type = st.selectbox("Owner Type", owner_options)

        fuel_options = sorted(train_df["fuel_type"].dropna().unique().tolist())
        fuel_type = st.selectbox("Fuel Type", fuel_options)

    with col2:
        distance_min = int(max(0, train_df["Distance"].min(skipna=True)))
        distance_max = int(train_df["Distance"].max(skipna=True))
        distance_default = int(train_df["Distance"].median(skipna=True))
        distance = st.slider(
            "Distance driven (km)",
            min_value=distance_min,
            max_value=max(distance_min + 1, distance_max),
            value=min(distance_default, max(distance_min + 1, distance_max)),
            step=1000,
        )

        year_min = int(train_df["manufacture_year"].min(skipna=True))
        year_max = int(train_df["manufacture_year"].max(skipna=True))
        manufacture_year = st.slider(
            "Manufacture year",
            min_value=year_min,
            max_value=year_max,
            value=int(train_df["manufacture_year"].median(skipna=True)),
        )

        age_min = int(max(0, train_df["Age of car"].min(skipna=True)))
        age_max = int(train_df["Age of car"].max(skipna=True))
        age = st.slider(
            "Age of car (years)",
            min_value=age_min,
            max_value=max(age_min + 1, age_max),
            value=int(train_df["Age of car"].median(skipna=True)),
        )

        engine_displacement = st.number_input(
            "Engine displacement",
            min_value=float(max(0, train_df["engine_displacement"].min(skipna=True))),
            max_value=float(train_df["engine_displacement"].max(skipna=True)),
            value=float(train_df["engine_displacement"].median(skipna=True)),
        )

    with col3:
        engine_power = st.number_input(
            "Engine power",
            min_value=float(max(0, train_df["engine_power"].min(skipna=True))),
            max_value=float(train_df["engine_power"].max(skipna=True)),
            value=float(train_df["engine_power"].median(skipna=True)),
        )

        audit_rating = st.number_input(
            "Vroom Audit Rating",
            min_value=int(train_df["Vroom Audit Rating"].min(skipna=True)),
            max_value=int(train_df["Vroom Audit Rating"].max(skipna=True)),
            value=int(train_df["Vroom Audit Rating"].median(skipna=True)),
        )

        body_options = ["Unknown"] + sorted(train_df["body_type"].dropna().unique().tolist())
        body_type = st.selectbox("Body Type", body_options)
        body_type = np.nan if body_type == "Unknown" else body_type

        transmission_options = sorted(train_df["transmission"].dropna().unique().tolist())
        transmission = st.selectbox("Transmission", transmission_options)

        door_values = sorted(train_df["door_count"].dropna().unique().astype(int).tolist())
        seat_values = sorted(train_df["seat_count"].dropna().unique().astype(int).tolist())
        door_count = st.selectbox("Door Count", door_values, index=min(0, len(door_values) - 1))
        seat_count = st.selectbox("Seat Count", seat_values, index=min(0, len(seat_values) - 1))

    input_row = pd.DataFrame(
        {
            "Maker": [maker],
            "model": [model_name],
            "Location": [location],
            "Distance": [distance],
            "Owner Type": [owner_type],
            "manufacture_year": [manufacture_year],
            "Age of car": [age],
            "engine_displacement": [engine_displacement],
            "engine_power": [engine_power],
            "body_type": [body_type],
            "Vroom Audit Rating": [audit_rating],
            "transmission": [transmission],
            "door_count": [door_count],
            "seat_count": [seat_count],
            "fuel_type": [fuel_type],
        }
    )

    if st.button("Predict Price", type="primary", use_container_width=True):
        try:
            prediction = float(model.predict(input_row)[0])
            lower = prediction * 0.95
            upper = prediction * 1.05

            st.success("Price prediction generated")
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated Price", f"₹{prediction:,.0f}")
            c2.metric("Fair Range — Lower", f"₹{lower:,.0f}")
            c3.metric("Fair Range — Upper", f"₹{upper:,.0f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Make sure the saved model is a pipeline that includes the same preprocessing "
                "used during training."
            )

with batch_tab:
    st.subheader("Batch prediction")
    st.write("Upload a CSV containing the same car features as the test dataset.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = normalize_columns(pd.read_csv(uploaded_file))

        required_missing = [col for col in FEATURES if col not in batch_df.columns]

        if required_missing:
            st.error("Missing columns: " + ", ".join(required_missing))
        else:
            prediction_input = batch_df[FEATURES].copy()

            if st.button("Run Batch Prediction", type="primary"):
                try:
                    preds = model.predict(prediction_input)
                    result = pd.DataFrame(
                        {
                            "ID": batch_df["ID"],
                            "Price": np.asarray(preds, dtype=float),
                        }
                    )
                    st.success(f"Generated {len(result):,} predictions.")
                    st.dataframe(result.head(20), use_container_width=True)

                    csv_bytes = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")

with insights_tab:
    st.subheader("Market insights")

    # Use native Streamlit charts as required by the capstone brief.
    age_price = (
        train_df.groupby("Age of car", as_index=False)["Price"]
        .median()
        .sort_values("Age of car")
    )
    st.write("Median price by car age")
    st.line_chart(age_price.set_index("Age of car"))

    distance_bins = pd.cut(train_df["Distance"], bins=10)
    distance_price = (
        train_df.groupby(distance_bins, observed=True)["Price"]
        .median()
        .reset_index()
    )
    distance_price["Distance Range"] = distance_price["Distance"].astype(str)
    st.write("Median price by distance range")
    st.bar_chart(distance_price.set_index("Distance Range")["Price"])

    maker_price = (
        train_df.groupby("Maker")["Price"]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )
    st.write("Top 15 makers by median price")
    st.bar_chart(maker_price)
