
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/amazon_reviews.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

SENTIMENT_ORDER = ["negative", "neutral", "positive"]


# ============================================================
# DATA PREPARATION
# ============================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    print("Loading dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required_columns = {"Review_text", "Rating"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    data = df[["Review_text", "Rating"]].copy()

    data["Review_text"] = (
        data["Review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["Rating"] = pd.to_numeric(
        data["Rating"], errors="coerce"
    )

    data = data[
        data["Review_text"].ne("")
        & data["Rating"].between(1, 5)
    ].copy()

    data["sentiment"] = data["Rating"].apply(
        lambda rating: (
            "negative" if rating <= 2
            else "neutral" if rating == 3
            else "positive"
        )
    )

    # Match the existing baseline's duplicate handling.
    data = data.drop_duplicates(
        subset=["Review_text"]
    ).reset_index(drop=True)

    counts = data["sentiment"].value_counts()

    if len(counts) != 3 or counts.min() < 2:
        raise ValueError(
            "All three sentiment classes must have "
            "at least two usable reviews."
        )

    print(f"Usable reviews: {len(data)}", flush=True)
    print("\nSentiment distribution:")
    print(counts.to_string(), flush=True)

    return data


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def main():
    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data["Review_text"],
        data["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["sentiment"],
    )

    print(f"\nTraining reviews: {len(X_train)}", flush=True)
    print(f"Testing reviews: {len(X_test)}", flush=True)

    # Keep the original baseline's TF-IDF settings.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Logistic Regression...", flush=True)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=SENTIMENT_ORDER,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=SENTIMENT_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=SENTIMENT_ORDER,
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=SENTIMENT_ORDER,
            zero_division=0,
        )
    )

    print(
        "\nConfusion matrix "
        "(rows = actual, columns = predicted):"
    )
    print(matrix)

    # ========================================================
    # SAVE MODEL AND METRICS
    # ========================================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "label_source": (
            "Star ratings: 1-2 negative, "
            "3 neutral, 4-5 positive"
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "total_usable_reviews": int(len(data)),
        "training_reviews": int(len(X_train)),
        "testing_reviews": int(len(X_test)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "sentiment_distribution": {
            str(label): int(count)
            for label, count in (
                data["sentiment"].value_counts().items()
            )
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_order": SENTIMENT_ORDER,
        "review_column": "Review_text",
        "rating_column": "Rating",
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed successfully.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
