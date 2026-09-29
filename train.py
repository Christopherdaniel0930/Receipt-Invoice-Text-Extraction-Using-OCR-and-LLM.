from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "train" / "categories.csv"
MODEL_PATH = BASE_DIR / "models" / "category_model.joblib"


# =========================================================
# Configuration
# =========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# =========================================================
# Load Dataset
# =========================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"text", "category"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    df = df.dropna(subset=["text", "category"])

    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    df = df[
        (df["text"] != "")
        & (df["category"] != "")
    ]

    return df


# =========================================================
# Train Model
# =========================================================

def train():
    print("=" * 60)
    print("EXPENSE CATEGORY CLASSIFIER TRAINING")
    print("=" * 60)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = load_dataset()

    print(f"\nDataset: {DATA_PATH}")
    print(f"Total samples: {len(df)}")

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    # -----------------------------------------------------
    # Check dataset
    # -----------------------------------------------------

    number_of_categories = df["category"].nunique()

    if number_of_categories < 2:
        raise ValueError(
            "At least 2 categories are required for training."
        )

    # -----------------------------------------------------
    # Train/Test Split
    # -----------------------------------------------------

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"TF-IDF feature count: "
        f"{X_train_tfidf.shape[1]}"
    )

    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_tfidf,
        y_train,
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    y_pred = model.predict(X_test_tfidf)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "categories": sorted(
            df["category"].unique().tolist()
        ),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nModel saved to:")
    print(MODEL_PATH)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    train()