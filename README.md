
# Amazon Customer Review Sentiment Analysis Using NLP

## 1. Project Overview

This project analyzes Amazon customer reviews using Natural Language Processing (NLP) and Machine Learning. It classifies customer reviews into three sentiment categories:

- **Positive:** Ratings of 4–5 stars
- **Neutral:** Ratings of 3 stars
- **Negative:** Ratings of 1–2 stars

The application provides an interactive Streamlit dashboard for analyzing individual reviews and multiple reviews from CSV files.

## 2. Objectives

- Process customer review text for sentiment analysis.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Predict sentiment for individual reviews.
- Analyze multiple reviews through CSV uploads.
- Visualize predicted sentiment distributions.
- Export prediction results as CSV files.
- Evaluate model performance using standard classification metrics.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Matplotlib
- Joblib

## 4. Project Structure

```text
Amazon_Sentiment_Analysis/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── amazon_reviews.csv
└── models/
    ├── sentiment_model.joblib
    └── evaluation_metrics.json
```

Additional experimental scripts and output files may also be present.

## 5. Dataset

The dataset is stored at `data/amazon_reviews.csv`.

The training script expects these columns:

- `Review_text`: Customer review text
- `Rating`: Numeric star rating from 1 to 5

Empty reviews, missing ratings, and ratings outside the valid range are excluded. Duplicate review text is removed before splitting the dataset.

The sentiment labels are derived from the star ratings. Therefore, the labels represent rating-based sentiment categories and may not always match the emotional meaning of the review text.

## 6. Methodology

### Step 1: Data Preparation

The dataset is loaded using Pandas. Review text is cleaned by removing empty values and trimming surrounding whitespace. Ratings are converted to numeric values.

### Step 2: Sentiment Labeling

Ratings are converted into three classes:

| Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

### Step 3: Train-Test Split

The dataset is divided into training and testing subsets using an 80:20 split. Stratification preserves the approximate class distribution in both subsets.

### Step 4: Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) converts review text into numerical features. Unigrams and bigrams help represent individual words and two-word phrases.

### Step 5: Model Training

Logistic Regression is trained on the TF-IDF features. Class weighting is used to help account for differences in class frequencies.

### Step 6: Evaluation

The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

### Step 7: Prediction Dashboard

Streamlit provides an interface for individual review predictions, batch CSV analysis, sentiment charts, and downloadable results.

## 7. Installation

Install Python and open a terminal in the project directory.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 8. Train the Model

Run:

```bash
python train_model.py
```

The training script saves the trained model and evaluation metrics in the `models` directory.

**Note:** Running the training script overwrites the existing saved model and metrics files.

## 9. Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## 10. Application Features

### Single Review Analysis

Enter a customer review and obtain a predicted sentiment. The dashboard may also display the model's estimated confidence.

### Batch Review Analysis

Upload a CSV containing a supported review-text column. The application predicts sentiment for each non-empty review.

### Visualizations

Bar charts and pie charts summarize the distribution of predicted positive, neutral, and negative reviews.

### Export Results

Download the batch prediction results as a CSV file for further analysis.

### Model Evaluation

View the saved evaluation metrics generated during training.

## 11. Experimental Results

The original baseline evaluation reported:

- **Test accuracy:** 81.85%
- **Macro F1-score:** Approximately 0.655
- **Negative-class F1-score:** Approximately 0.756
- **Neutral-class F1-score:** Approximately 0.303
- **Positive-class F1-score:** Approximately 0.905

These results apply to the original baseline evaluation. Results may differ when the dataset, split, or model configuration changes.

The neutral class has substantially lower F1-score than the positive class. This indicates that the model struggles to identify neutral reviews reliably, despite its overall accuracy.

## 12. Limitations

- Sentiment labels are derived from star ratings rather than manually annotated review text.
- Three-star reviews may express mixed or unclear opinions.
- Class imbalance can affect performance across sentiment categories.
- TF-IDF does not fully capture context, sarcasm, or complex language.
- Model confidence is not a guarantee of prediction correctness.
- Predictions for uploaded reviews may not match human judgments.

## 13. Future Enhancements

- Compare additional text-classification algorithms.
- Tune model parameters using validation data.
- Investigate neutral-class misclassification.
- Explore word and character TF-IDF features.
- Add more detailed evaluation charts.
- Investigate transformer-based sentiment models.
- Improve deployment monitoring and input validation.

## 14. Conclusion

The project demonstrates an end-to-end sentiment analysis workflow using NLP and Machine Learning. It includes data preparation, feature extraction, model training, evaluation, interactive prediction, data visualization, and CSV export.

The results demonstrate the usefulness of TF-IDF and Logistic Regression as a baseline approach while highlighting the need for further work on neutral sentiment classification.

## 15. Academic Use

This project is intended for academic learning and demonstration. Dataset sources, licensing conditions, and attribution requirements should be verified and acknowledged in the final project report.
