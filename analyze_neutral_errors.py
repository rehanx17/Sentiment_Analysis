
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("models/neutral_error_samples.csv")
OUTPUT_PATH = Path("models/neutral_error_summary.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Error samples not found: {INPUT_PATH}\n"
            "Run inspect_neutral.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "review",
        "actual_sentiment",
        "predicted_sentiment",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Analyze text length without printing review contents.
    df["review"] = df["review"].fillna("").astype(str)
    df["word_count"] = df["review"].str.split().str.len()
    df["character_count"] = df["review"].str.len()

    print("\n========== NEUTRAL ERROR ANALYSIS ==========")
    print(f"Sampled misclassified reviews: {len(df)}")

    print("\nPredicted sentiment for actual neutral reviews:")
    print(df["predicted_sentiment"].value_counts().to_string())

    print("\nReview length statistics:")
    print(
        df.groupby("predicted_sentiment")[
            ["word_count", "character_count"]
        ].median().round(1).to_string()
    )

    bins = [-1, 5, 10, 20, 50, float("inf")]
    labels = [
        "0-5 words",
        "6-10 words",
        "11-20 words",
        "21-50 words",
        "Over 50 words",
    ]

    df["length_group"] = pd.cut(
        df["word_count"],
        bins=bins,
        labels=labels,
    )

    print("\nError sample counts by review length:")
    print(
        pd.crosstab(
            df["length_group"],
            df["predicted_sentiment"],
            margins=True,
        ).to_string()
    )

    # Save only aggregate statistics, not review text.
    summary = (
        df.groupby(
            ["predicted_sentiment", "length_group"],
            observed=False,
        )
        .size()
        .reset_index(name="sample_count")
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSummary saved to: {OUTPUT_PATH}")
    print(
        "No model, dataset, or Streamlit app files were changed."
    )


if __name__ == "__main__":
    main()
