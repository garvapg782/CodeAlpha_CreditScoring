"""
CodeAlpha Machine Learning Internship - Task 1
Credit Scoring Model (German Credit dataset)

Predicts whether an applicant is a HIGH credit risk (likely to default) using
Logistic Regression, Decision Tree and Random Forest, with engineered financial
features and Precision / Recall / F1 / ROC-AUC evaluation.
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "data/german_credit.csv"
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    """Load the local copy of the dataset (falls back to downloading it)."""
    df = pd.read_csv(DATA_PATH if os.path.exists(DATA_PATH) else DATA_URL)
    # Original column: credit_risk 1 = good, 0 = bad.
    # Flip so that 1 = HIGH RISK (the event we want to catch).
    df["high_risk"] = 1 - df["credit_risk"]
    return df.drop(columns="credit_risk")


def engineer_features(df):
    """Create financial-behaviour features from the raw columns."""
    df = df.copy()
    df["monthly_payment"] = df["amount"] / df["duration"]               # repayment burden
    df["payment_to_income_proxy"] = df["monthly_payment"] * df["installment_rate"]
    df["amount_per_year_of_age"] = df["amount"] / df["age"]
    df["log_amount"] = np.log1p(df["amount"])
    df["long_loan"] = (df["duration"] > 24).astype(int)
    df["young_borrower"] = (df["age"] < 25).astype(int)
    return df


def explore(df):
    print("Shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("\nClass distribution (1 = high risk):")
    print(df["high_risk"].value_counts())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.countplot(x="high_risk", data=df, palette=["#4C9F70", "#D1495B"], ax=axes[0])
    axes[0].set_xticklabels(["Low risk", "High risk"])
    axes[0].set_title("Class Distribution")
    sns.boxplot(x="high_risk", y="duration", data=df, palette=["#4C9F70", "#D1495B"], ax=axes[1])
    axes[1].set_xticklabels(["Low risk", "High risk"])
    axes[1].set_title("Loan Duration (months) by Risk")
    sns.boxplot(x="high_risk", y="amount", data=df, palette=["#4C9F70", "#D1495B"], ax=axes[2])
    axes[2].set_xticklabels(["Low risk", "High risk"])
    axes[2].set_title("Credit Amount by Risk")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
def build_preprocessor(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def build_models(X):
    def pipe(clf):
        return Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])

    # class_weight="balanced" because only 30% of applicants are high risk
    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                       random_state=RANDOM_STATE)),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                                     class_weight="balanced",
                                                     random_state=RANDOM_STATE)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                     random_state=RANDOM_STATE)),
    }


def evaluate(models, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, probas = [], {}
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        probas[name] = proba
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1-Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "CV ROC-AUC (5-fold)": cv_auc,
        })
    return pd.DataFrame(rows).set_index("Model").round(4), probas


def plot_roc(probas, y_test):
    plt.figure(figsize=(6, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Credit Scoring")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
    plt.close()


def tune_random_forest(X, X_train, y_train):
    pipe = Pipeline([("prep", build_preprocessor(X)),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))])
    grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [5, 8, None],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    search = GridSearchCV(pipe, grid, scoring="roc_auc", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    search.fit(X_train, y_train)
    print("\nBest Random Forest params:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def best_threshold(model, X_train, y_train):
    """Pick the decision threshold that maximises F1 on out-of-fold predictions
    (avoids tuning on the test set)."""
    from sklearn.model_selection import cross_val_predict
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def final_report(model, threshold, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= threshold).astype(int)

    print("\n=== Tuned Random Forest: Test Set (threshold 0.50) ===")
    print(classification_report(y_test, pred_default, target_names=["Low risk", "High risk"]))
    print(f"=== Same model with tuned threshold {threshold:.2f} ===")
    print(classification_report(y_test, pred_tuned, target_names=["Low risk", "High risk"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    cm = confusion_matrix(y_test, pred_tuned)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low risk", "High risk"], yticklabels=["Low risk", "High risk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()


def feature_importance(model):
    names = model.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=imp.values, y=imp.index, color="#3A7CA5")
    plt.title("Top 12 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print("\nTop 8 features:\n", imp.head(8).round(4))


def main():
    df = load_data()
    explore(df)
    df = engineer_features(df)

    X, y = df.drop(columns="high_risk"), df["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results, probas = evaluate(build_models(X), X_train, X_test, y_train, y_test)
    print("\n=== Model Comparison (Test Set, threshold 0.50) ===")
    print(results.to_string())
    results.to_csv(f"{OUT_DIR}/model_comparison.csv")
    plot_roc(probas, y_test)

    best = tune_random_forest(X, X_train, y_train)
    thr = best_threshold(best, X_train, y_train)
    final_report(best, thr, X_test, y_test)
    feature_importance(best)

    joblib.dump({"model": best, "threshold": thr}, f"{OUT_DIR}/credit_model.joblib")
    print(f"\nSaved model and plots to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
