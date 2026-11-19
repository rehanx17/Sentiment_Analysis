
import json
import time
from pathlib import Path

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


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/amazon_reviews.csv")
MODEL_DIR = Path("models")
RESULTS_PATH = MODEL_DIR / "neutral_tuning_results.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_OF_REMAINDER = 0.25

SENTIMENT_ORDER = ["negative", "neutral", "positive"]


# ============================================================
# LOAD AND VALIDATE DATA
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
            f"Missing required columns: {sorted(missing)}. "
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

    # Keep duplicate handling consistent with the baseline.
    data = data.drop_duplicates(
        subset=["Review_text"]
    ).reset_index(drop=True)

    counts = data["sentiment"].value_counts()

    if not set(SENTIMENT_ORDER).issubset(counts.index):
        raise ValueError(
            "The dataset must contain negative, neutral, "
            "and positive examples."
        )

    if counts.min() < 3:
        raise ValueError(
            "Each sentiment class needs at least three reviews."
        )

    print(f"Usable reviews: {len(data)}", flush=True)
    print("\nSentiment distribution:")
    print(counts.to_string(), flush=True)

    return data


# ============================================================
# EVALUATION HELPER
# ============================================================

def evaluate_model(model, X, y):
    predictions = model.predict(X)

    report = classification_report(
        y,
        predictions,
        labels=SENTIMENT_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=SENTIMENT_ORDER,
    )

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=SENTIMENT_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                predictions,
                labels=SENTIMENT_ORDER,
                average="weighted",
                zero_division=0,
            )
        ),
        "neutral_f1": float(
            report["neutral"]["f1-score"]
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_order": SENTIMENT_ORDER,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    data = load_dataset()

    X = data["Review_text"]
    y = data["sentiment"]

    # Reserve 20% for the final test.
    X_remaining, X_test, y_remaining, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Split the remaining 80% into 60% train and 20% validation.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_remaining,
        y_remaining,
        test_size=VALIDATION_SIZE_OF_REMAINDER,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    print("\nDataset split:", flush=True)
    print(f"Training reviews:   {len(X_train)}", flush=True)
    print(f"Validation reviews: {len(X_valid)}", flush=True)
    print(f"Final test reviews: {len(X_test)}", flush=True)

    # --------------------------------------------------------
    # FIT TF-IDF ON TRAINING TEXT ONLY
    # --------------------------------------------------------

    print("\nBuilding TF-IDF features...", flush=True)
    start = time.time()

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

    print(
        f"TF-IDF finished in {time.time() - start:.1f} seconds.",
        flush=True,
    )
    print(
        f"Number of features: {X_train_tfidf.shape[1]}",
        flush=True,
    )

    # --------------------------------------------------------
    # CANDIDATE CONFIGURATIONS
    # --------------------------------------------------------

    candidates = [
        {
            "name": "balanced_C1",
            "C": 1.0,
            "class_weight": "balanced",
        },
        {
            "name": "balanced_C2",
            "C": 2.0,
            "class_weight": "balanced",
        },
        {
            "name": "neutral_1_5_C1",
            "C": 1.0,
            "class_weight": {
                "negative": 1.0,
                "neutral": 1.5,
                "positive": 1.0,
            },
        },
        {
            "name": "neutral_2_C1",
            "C": 1.0,
            "class_weight": {
                "negative": 1.0,
                "neutral": 2.0,
                "positive": 1.0,
            },
        },
        {
            "name": "neutral_3_C1",
            "C": 1.0,
            "class_weight": {
                "negative": 1.0,
                "neutral": 3.0,
                "positive": 1.0,
            },
        },
    ]

    validation_results = []
    best_model = None
    best_config = None
    best_macro_f1 = -1.0

    # --------------------------------------------------------
    # TRAIN AND SELECT USING VALIDATION MACRO F1
    # --------------------------------------------------------

    for index, config in enumerate(candidates, start=1):
        print(
            f"\n[{index}/{len(candidates)}] "
            f"Training {config['name']}...",
            flush=True,
        )

        model = LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

        start = time.time()
        model.fit(X_train_tfidf, y_train)

        validation = evaluate_model(
            model, X_valid_tfidf, y_valid
        )

        elapsed = time.time() - start

        result = {
            "name": config["name"],
            "C": config["C"],
            "class_weight": config["class_weight"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_weighted_f1": validation["weighted_f1"],
            "validation_neutral_f1": validation["neutral_f1"],
        }

        validation_results.append(result)

        print(
            f"Time: {elapsed:.1f}s | "
            f"Accuracy: {validation['accuracy']:.4f} | "
            f"Macro F1: {validation['macro_f1']:.4f} | "
            f"Neutral F1: {validation['neutral_f1']:.4f}",
            flush=True,
        )

        # The final test set is NOT used for selection.
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_model = model
            best_config = config

    # --------------------------------------------------------
    # EVALUATE THE SELECTED MODEL ON THE HELD-OUT TEST SET
    # --------------------------------------------------------

    print("\nBest validation configuration:", flush=True)
    print(json.dumps(best_config, indent=4), flush=True)

    print("\nPreparing final test features...", flush=True)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Evaluating selected model on final test set...", flush=True)

    final_test = evaluate_model(
        best_model, X_test_tfidf, y_test
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Accuracy: {final_test['accuracy']:.4f}")
    print(f"Macro F1: {final_test['macro_f1']:.4f}")
    print(f"Weighted F1: {final_test['weighted_f1']:.4f}")
    print(f"Neutral F1: {final_test['neutral_f1']:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            best_model.predict(X_test_tfidf),
            labels=SENTIMENT_ORDER,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "(rows = actual, columns = predicted):"
    )
    print(final_test["confusion_matrix"], flush=True)

    # --------------------------------------------------------
    # SAVE RESULTS ONLY; DO NOT REPLACE THE MODEL
    # --------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "Validation-based neutral sentiment tuning",
        "random_state": RANDOM_STATE,
        "split": {
            "training_reviews": int(len(X_train)),
            "validation_reviews": int(len(X_valid)),
            "test_reviews": int(len(X_test)),
        },
        "selection_metric": "validation_macro_f1",
        "best_validation_configuration": best_config,
        "validation_results": validation_results,
        "final_test_results": final_test,
        "notes": [
            "The final test set was not used to choose the configuration.",
            "The original deployed model was not changed.",
            "Compare with the original baseline cautiously because "
            "the data split and TF-IDF settings differ.",
        ],
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(
        f"\nResults saved to: {RESULTS_PATH}",
        flush=True,
    )
    print(
        "Existing sentiment_model.joblib and app.py were not changed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
