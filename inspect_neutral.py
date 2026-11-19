
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/amazon_reviews.csv")
OUTPUT_PATH = Path("models/neutral_error_samples.csv")

RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]


def main():
    print("Loading dataset...", flush=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)

    data = df[["Review_text", "Rating"]].copy()
    data["Review_text"] = (
        data["Review_text"].fillna("").astype(str).str.strip()
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

    data = data.drop_duplicates(
        subset=["Review_text"]
    ).reset_index(drop=True)

    X_train, X_remaining, y_train, y_remaining = (
        train_test_split(
            data["Review_text"],
            data["sentiment"],
            test_size=0.40,
            random_state=RANDOM_STATE,
            stratify=data["sentiment"],
        )
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    # Fit text features on training data only.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    print("Training diagnostic model...", flush=True)
    model.fit(X_train_tfidf, y_train)

    # Validation data is used only to report a diagnostic score.
    valid_predictions = model.predict(X_valid_tfidf)
    print(
        "\nValidation confusion matrix "
        "(rows = actual, columns = predicted):"
    )
    print(confusion_matrix(
        y_valid, valid_predictions, labels=LABELS
    ))

    # Inspect the held-out test errors.
    predictions = model.predict(X_test_tfidf)

    diagnostic = pd.DataFrame({
        "review": X_test.to_numpy(),
        "actual_sentiment": y_test.to_numpy(),
        "predicted_sentiment": predictions,
    })

    neutral_errors = diagnostic[
        (diagnostic["actual_sentiment"] == "neutral")
        & (diagnostic["predicted_sentiment"] != "neutral")
    ].copy()

    neutral_errors["error_type"] = (
        "Actual neutral predicted as "
        + neutral_errors["predicted_sentiment"]
    )

    # Sample a maximum of 100 errors, reproducibly.
    sample = neutral_errors.sample(
        n=min(100, len(neutral_errors)),
        random_state=RANDOM_STATE,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    print(f"\nActual neutral reviews in test set: "
          f"{int((y_test == 'neutral').sum())}")
    print(f"Neutral reviews misclassified: {len(neutral_errors)}")
    print(f"Error samples saved to: {OUTPUT_PATH}")
    print(
        "\nOpen the CSV and inspect the review text and "
        "predicted sentiment. Do not share any private "
        "or personally identifying review information."
    )


if __name__ == "__main__":
    main()
