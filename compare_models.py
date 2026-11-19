
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ---------------- CONFIGURATION ----------------

DATA_PATH = Path("data/amazon_reviews.csv")
MODEL_DIR = Path("models")
RESULTS_PATH = MODEL_DIR / "model_comparison.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

SENTIMENT_ORDER = ["negative", "neutral", "positive"]


# ---------------- MAIN FUNCTION ----------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Loading dataset...", flush=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)

    required_columns = {"Review_text", "Rating"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Required columns are Review_text and Rating. "
            f"Found: {df.columns.tolist()}"
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

    data = data.drop_duplicates(
        subset=["Review_text"]
    ).reset_index(drop=True)

    counts = data["sentiment"].value_counts()

    if len(counts) != 3 or counts.min() < 2:
        raise ValueError(
            "The dataset must contain all three sentiment "
            "classes with at least two reviews per class."
        )

    print(f"Usable reviews: {len(data)}", flush=True)
    print("\nSentiment distribution:", flush=True)
    print(counts.to_string(), flush=True)

    # ---------------- TRAIN / TEST SPLIT ----------------

    X_train, X_test, y_train, y_test = train_test_split(
        data["Review_text"],
        data["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["sentiment"],
    )

    print(f"\nTraining reviews: {len(X_train)}", flush=True)
    print(f"Testing reviews: {len(X_test)}", flush=True)

    # ---------------- TF-IDF: FIT ONLY ONCE ----------------

    print("\nBuilding TF-IDF features...", flush=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
        dtype="float32",
    )

    start = time.time()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"TF-IDF complete: {X_train_tfidf.shape[1]} features "
        f"in {time.time() - start:.1f} seconds.",
        flush=True,
    )

    # ---------------- MODELS ----------------

    models = {
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.5
        ),
        "Linear SVM (balanced)": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=2000,
        ),
    }

    results = {}

    # ---------------- TRAIN AND EVALUATE ----------------

    for index, (name, model) in enumerate(
        models.items(), start=1
    ):
        print(
            f"\n[{index}/{len(models)}] Training {name}...",
            flush=True,
        )

        start = time.time()

        model.fit(X_train_tfidf, y_train)

        print(
            f"Training finished in {time.time() - start:.1f} seconds.",
            flush=True,
        )

        predictions = model.predict(X_test_tfidf)

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

        results[name] = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "macro_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    labels=SENTIMENT_ORDER,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    labels=SENTIMENT_ORDER,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "class_order": SENTIMENT_ORDER,
        }

        print(
            f"Accuracy: {results[name]['accuracy']:.4f}",
            flush=True,
        )
        print(
            f"Macro F1: {results[name]['macro_f1']:.4f}",
            flush=True,
        )
        print(
            f"Neutral F1: "
            f"{report['neutral']['f1-score']:.4f}",
            flush=True,
        )

        print(
            "Confusion matrix (actual rows, predicted columns):",
            flush=True,
        )
        print(matrix, flush=True)

    # ---------------- SAVE RESULTS ----------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    summary = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Macro F1": round(metrics["macro_f1"], 4),
            "Weighted F1": round(metrics["weighted_f1"], 4),
            "Neutral F1": round(
                metrics["classification_report"]["neutral"][
                    "f1-score"
                ],
                4,
            ),
        }
        for name, metrics in results.items()
    ])

    summary = summary.sort_values(
        "Macro F1", ascending=False
    )

    print("\n========== MODEL COMPARISON ==========", flush=True)
    print(summary.to_string(index=False), flush=True)

    print(
        f"\nDetailed results saved to: {RESULTS_PATH}",
        flush=True,
    )
    print(
        "\nComparison complete. The existing deployed model "
        "has not been changed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
