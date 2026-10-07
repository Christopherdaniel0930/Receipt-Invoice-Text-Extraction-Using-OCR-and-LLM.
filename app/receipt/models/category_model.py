from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path("models/receipt/category_model.joblib")


def train_category_model(texts, labels, model_path=MODEL_PATH):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, model_path)


def predict_category(text, model_path=MODEL_PATH):
    bundle = joblib.load(model_path)
    X = bundle["vectorizer"].transform([text])
    model = bundle["model"]
    return str(model.predict(X)[0]), float(model.predict_proba(X).max())
