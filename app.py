
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "evaluation_metrics.json"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#EF4444",
    "neutral": "#F59E0B",
    "positive": "#10B981"
}


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


# ---------------- HELPER FUNCTIONS ----------------
def predict_sentiment(model, texts):
    predictions = model.predict(texts)
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)

    results = []

    for index, sentiment in enumerate(predictions):
        row = {
            "Sentiment": str(sentiment).capitalize()
        }

        if probabilities is not None:
            classes = list(model.classes_)
            row["Confidence (%)"] = round(
                float(max(probabilities[index])) * 100, 2
            )

        results.append(row)

    return results


def find_review_column(columns):
    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in columns
    }

    candidates = [
        "review_text",
        "review",
        "review_body",
        "review_content",
        "text",
        "content",
        "review_header"
    ]

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📊 Review Analyzer")
    st.write("NLP-powered Amazon review sentiment classification.")

    st.divider()
    st.subheader("Project Features")
    st.markdown(
        """
        - Sentiment prediction
        - Review dataset analysis
        - Sentiment distribution charts
        - CSV upload
        - Downloadable predictions
        - Model evaluation summary
        """
    )

    st.divider()
    st.caption("Model: TF-IDF + Logistic Regression")


# ---------------- MAIN DASHBOARD ----------------
st.title("Amazon Customer Review Sentiment Analysis")
st.write(
    "Analyze customer reviews using Natural Language Processing "
    "and a trained machine learning model."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

# Show model accuracy from the saved evaluation file if available.
accuracy = metrics.get("accuracy", metrics.get("test_accuracy"))

if accuracy is not None:
    try:
        accuracy_value = float(accuracy)
        if accuracy_value > 1:
            accuracy_value /= 100
        st.metric("Model Test Accuracy", f"{accuracy_value:.2%}")
    except (ValueError, TypeError):
        st.info("Model evaluation metrics are available in the models folder.")

st.divider()

# ---------------- SINGLE REVIEW PREDICTION ----------------
st.subheader("1. Analyze a Single Review")

review_text = st.text_area(
    "Enter an Amazon customer review",
    placeholder="Example: The product quality is excellent and delivery was fast.",
    height=120
)

if st.button("Analyze Review", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([review_text])[0]
        sentiment = str(prediction).lower()

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review_text])[0]
            confidence = float(max(probabilities)) * 100
        else:
            confidence = None

        color = SENTIMENT_COLORS.get(sentiment, "#64748B")

        st.markdown(
            f"""
            <div style="padding:18px;border-radius:10px;
                        background-color:{color};color:white;">
                <h3 style="margin:0;">Predicted sentiment:
                {sentiment.capitalize()}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence is not None:
            st.metric("Model confidence", f"{confidence:.2f}%")

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

st.divider()

# ---------------- CSV DATASET ANALYSIS ----------------
st.subheader("2. Analyze a CSV File")

st.write(
    "Upload a CSV containing a review-text column. "
    "The dashboard will predict sentiment for each valid review."
)

uploaded_file = st.file_uploader(
    "Upload review dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.write("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)

        review_column = find_review_column(df.columns)

        if review_column is None:
            st.error(
                "Could not identify the review-text column. "
                "Use a column named Review_text, Review Text, "
                "Review, Text, or Content."
            )
        else:
            st.success(f"Review column detected: {review_column}")

            valid_reviews = df[review_column].fillna("").astype(str).str.strip()
            valid_mask = valid_reviews.ne("")

            if not valid_mask.any():
                st.warning("No non-empty reviews were found in this file.")
            else:
                st.caption(
                    f"{int(valid_mask.sum()):,} non-empty reviews found."
                )

                max_rows = st.number_input(
                    "Maximum reviews to analyze",
                    min_value=1,
                    max_value=int(valid_mask.sum()),
                    value=min(5000, int(valid_mask.sum())),
                    step=500
                )

                if st.button("Analyze Uploaded Dataset", type="primary"):
                    work_df = df.loc[valid_mask].head(int(max_rows)).copy()
                    texts = (
                        work_df[review_column]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )

                    with st.spinner("Analyzing reviews..."):
                        predicted = model.predict(texts)

                        work_df["Predicted Sentiment"] = [
                            str(item).capitalize() for item in predicted
                        ]

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(texts)
                            work_df["Confidence (%)"] = [
                                round(float(max(row)) * 100, 2)
                                for row in probabilities
                            ]

                    st.session_state["analysis_results"] = work_df
                    st.success(
                        f"Analysis completed for {len(work_df):,} reviews."
                    )

    except Exception as error:
        st.error(f"Could not read or analyze this CSV: {error}")

# ---------------- DISPLAY ANALYSIS RESULTS ----------------
results = st.session_state.get("analysis_results")

if results is not None and not results.empty:
    st.divider()
    st.subheader("3. Sentiment Analysis Results")

    counts = (
        results["Predicted Sentiment"]
        .value_counts()
        .reindex(["Negative", "Neutral", "Positive"], fill_value=0)
    )

    total = len(results)
    negative_count = int(counts["Negative"])
    neutral_count = int(counts["Neutral"])
    positive_count = int(counts["Positive"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Reviews Analyzed", f"{total:,}")
    col2.metric("Positive", f"{positive_count:,}")
    col3.metric("Neutral", f"{neutral_count:,}")
    col4.metric("Negative", f"{negative_count:,}")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Sentiment Distribution")

        fig, ax = plt.subplots()
        chart_colors = [
            SENTIMENT_COLORS["negative"],
            SENTIMENT_COLORS["neutral"],
            SENTIMENT_COLORS["positive"]
        ]

        ax.bar(
            counts.index,
            counts.values,
            color=chart_colors
        )
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Predicted Sentiment Counts")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Sentiment Proportion")

        fig, ax = plt.subplots()
        nonzero_counts = counts[counts > 0]

        ax.pie(
            nonzero_counts.values,
            labels=nonzero_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=[
                SENTIMENT_COLORS[label.lower()]
                for label in nonzero_counts.index
            ]
        )
        ax.set_title("Percentage of Predicted Sentiments")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Prediction Table")
    st.dataframe(results, use_container_width=True)

    csv_data = results.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download Predictions as CSV",
        data=csv_data,
        file_name="amazon_sentiment_predictions.csv",
        mime="text/csv"
    )

    st.caption(
        "The charts show the model's predictions, not verified "
        "ground-truth sentiment labels."
    )

st.divider()

# ---------------- MODEL EVALUATION ----------------
st.subheader("4. Model Evaluation")

if metrics:
    st.write(
        "These metrics describe evaluation on the held-out test set "
        "used during model training."
    )

    st.json(metrics)
else:
    st.info(
        "Evaluation metrics file was not found or contains no data. "
        "Check models/evaluation_metrics.json."
    )

st.divider()
st.caption(
    "Academic project | Natural Language Processing | "
    "TF-IDF and Logistic Regression"
)
