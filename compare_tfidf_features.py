
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
from sklearn.pipeline import FeatureUnion


DATA_PATH = Path("data/amazon_reviews.csv")
RESULT_PATH = Path("models/tfidf_feature_comparison.json")

RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required = {"Review_text", "Rating"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required}; found {df.columns.tolist()}"
        )

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
        lambda r: "negative" if r <= 2
        else "neutral" if r == 3
        else "positive"
    )
    data = data.drop_duplicates(
        subset=["Review_text"]
    ).reset_index(drop=True)

    X_train, X_rest, y_train, y_rest = train_test_split(
        data["Review_text"],
        data["sentiment"],
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=data["sentiment"],
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_rest,
    )

    print(
        f"Training: {len(X_train)}, validation: {len(X_valid)}, "
        f"test: {len(X_test)}",
        flush=True,
    )

    feature_sets = {
        "word_tfidf": TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=3,
            max_features=30000,
            sublinear_tf=True,
        ),
        "word_plus_char_tfidf": FeatureUnion([
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
        ]),
    }

    results = {}
    best_name = None
    best_macro = -1
    best_model = None
    best_vectorizer = None

    for name, vectorizer in feature_sets.items():
        print(f"\nFitting features: {name}", flush=True)
        start = time.time()

        Xtr = vectorizer.fit_transform(X_train)
        Xv = vectorizer.transform(X_valid)

        print(
            f"Features ready: {Xtr.shape[1]} in "
            f"{time.time() - start:.1f}s",
            flush=True,
        )

        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

        print(f"Training model: {name}", flush=True)
        model.fit(Xtr, y_train)
        pred = model.predict(Xv)

        report = classification_report(
            y_valid,
            pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        score = macro_f1(y_valid, pred)

        results[name] = {
            "validation_accuracy": float(
                accuracy_score(y_valid, pred)
            ),
            "validation_macro_f1": float(score),
            "validation_neutral_f1": float(
                report["neutral"]["f1-score"]
            ),
            "validation_report": report,
        }

        print(
            f"Validation accuracy: {results[name]['validation_accuracy']:.4f}",
            flush=True,
        )
        print(
            f"Validation macro F1: {score:.4f}",
            flush=True,
        )
        print(
            f"Validation neutral F1: "
            f"{results[name]['validation_neutral_f1']:.4f}",
            flush=True,
        )

        if score > best_macro:
            best_macro = score
            best_name = name
            best_model = model
            best_vectorizer = vectorizer

    # Evaluate the selected feature configuration once on test data.
    print(f"\nSelected using validation macro F1: {best_name}")
    Xt = best_vectorizer.transform(X_test)
    test_pred = best_model.predict(Xt)

    test_report = classification_report(
        y_test,
        test_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    test_results = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "macro_f1": float(macro_f1(y_test, test_pred)),
        "neutral_f1": float(test_report["neutral"]["f1-score"]),
        "classification_report": test_report,
        "confusion_matrix": confusion_matrix(
            y_test, test_pred, labels=LABELS
        ).tolist(),
        "class_order": LABELS,
    }

    print("\n========== SELECTED MODEL TEST RESULTS ==========")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Macro F1: {test_results['macro_f1']:.4f}")
    print(f"Neutral F1: {test_results['neutral_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=LABELS,
            zero_division=0,
        )
    )
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(test_results["confusion_matrix"])

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "validation_results": results,
            "selected_features": best_name,
            "selection_metric": "validation_macro_f1",
            "test_results": test_results,
            "notes": [
                "Feature selection used validation macro F1.",
                "The final test set was not used to select features.",
                "No existing model or app files were changed.",
            ],
        }, f, indent=4)

    print(f"\nResults saved to: {RESULT_PATH}", flush=True)
    print("Existing model and Streamlit app remain unchanged.")


if __name__ == "__main__":
    main()
