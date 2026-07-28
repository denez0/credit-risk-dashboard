"""
Credit Risk Model - Interactive Dashboard
Author: Denys Yakovliev
Purpose: End-to-end credit risk classification with EDA, CV, and visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, 
    confusion_matrix
)

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("📊 Credit Risk Model Dashboard")

# ============ EDA SECTION ============
st.header("🔍 Exploratory Data Analysis")

@st.cache_data
def generate_data():
    """Generate synthetic credit data with EDA insights."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        flip_y=0.03,
        random_state=42
    )
    
    feature_names = [
        "income_ratio", "debt_to_income", "credit_utilization",
        "late_payments_30d", "late_payments_60d", "credit_age_months",
        "num_credit_lines", "inquiry_count_6m", "bankruptcy_flag",
        "employment_years", "residential_stability", "loan_amount_ratio",
        "interest_rate", "collateral_value", "previous_default",
        "feature_16", "feature_17", "feature_18", "feature_19", "feature_20"
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df["default"] = y
    return df, feature_names

df, feature_names = generate_data()

# EDA Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["default"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
    ax.set_xticklabels(["Non-Default", "Default"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title(f"Default Rate: {df['default'].mean():.2%}")
    st.pyplot(fig)

with col2:
    st.subheader("Key Feature Distributions by Default Status")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    key_features = ["income_ratio", "debt_to_income", "credit_utilization", "late_payments_30d"]
    
    for i, feat in enumerate(key_features):
        row, col = i // 2, i % 2
        for default_val in [0, 1]:
            df[df["default"] == default_val][feat].hist(
                bins=30, alpha=0.5, label=f"Default={default_val}", ax=axes[row, col]
            )
        axes[row, col].set_title(feat)
        axes[row, col].legend()
    plt.tight_layout()
    st.pyplot(fig)

# EDA Statistics
with st.expander("📊 Detailed Summary Statistics"):
    st.dataframe(df.describe())
    
    st.subheader("Feature Correlation with Target")
    corr = df.corr()["default"].sort_values(ascending=False)
    st.dataframe(corr.to_frame("Correlation"))

# ============ TRAIN MODELS WITH CROSS-VALIDATION ============
st.header("🧠 Model Training & Cross-Validation")

@st.cache_resource
def train_models():
    """Train models with cross-validation."""
    X = df.drop("default", axis=1)
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Logistic Regression
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
        max_iter=1000
    )
    
    # Cross-validation
    lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="roc_auc")
    lr.fit(X_train, y_train)
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
    rf.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return {
        "X_test": X_test,
        "y_test": y_test,
        "lr_model": lr,
        "rf_model": rf,
        "lr_cv_scores": lr_cv_scores,
        "rf_cv_scores": rf_cv_scores,
        "feature_importance": feature_importance
    }

data = train_models()
X_test = data["X_test"]
y_test = data["y_test"]
lr_model = data["lr_model"]
rf_model = data["rf_model"]
lr_cv_scores = data["lr_cv_scores"]
rf_cv_scores = data["rf_cv_scores"]
feature_importance = data["feature_importance"]

# ============ SIDEBAR ============
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
threshold = st.sidebar.slider("Probability Threshold", 0.1, 0.9, 0.5, 0.05)

# Select model
if model_choice == "Logistic Regression":
    model = lr_model
    cv_mean = lr_cv_scores.mean()
    cv_stdev = lr_cv_scores.std()
else:
    model = rf_model
    cv_mean = rf_cv_scores.mean()
    cv_stdev = rf_cv_scores.std()

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# ============ METRICS ============
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
gini = 2 * auc - 1
cm = confusion_matrix(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("AUC", f"{auc:.3f}")
col3.metric("Gini", f"{gini:.3f}")
col4.metric("CV AUC", f"{cv_mean:.3f}", f"±{cv_stdev:.3f}")

# ============ CHARTS ============
left, right = st.columns(2)

# Confusion Matrix
with left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ROC Curve
with right:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# Feature Importance
st.subheader("Top 10 Feature Importances (Random Forest)")
fig, ax = plt.subplots(figsize=(10, 5))
top_features = feature_importance.head(10)
ax.barh(top_features["feature"], top_features["importance"])
ax.set_xlabel("Importance")
ax.invert_yaxis()
st.pyplot(fig)

# ============ CROSS-VALIDATION DETAIL ============
with st.expander("📈 Cross-Validation Details"):
    st.write("**5-Fold Cross-Validation AUC Scores:**")
    
    cv_df = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "Logistic Regression": lr_cv_scores,
        "Random Forest": rf_cv_scores
    })
    st.dataframe(cv_df)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([lr_cv_scores, rf_cv_scores], label=["Logistic Regression", "Random Forest"])
    ax.set_ylabel("AUC Score")
    ax.set_title("Cross-Validation AUC Distribution")
    st.pyplot(fig)

# ============ PREDICT SINGLE ============
st.subheader("🔮 Test a Single Application")

if st.button("🎲 Pick Random Sample"):
    idx = np.random.randint(0, len(X_test))
    sample = X_test.iloc[idx]
    proba = model.predict_proba(sample.values.reshape(1, -1))[0][1]
    pred = "⚠️ Default Risk" if proba >= threshold else "✅ Low Risk"
    
    st.write("**Application Features:**")
    st.dataframe(pd.DataFrame(sample).T)
    st.metric("Default Probability", f"{proba:.1%}", delta=pred)

# ============ BUSINESS INTERPRETATION ============
with st.expander("📖 Business Interpretation"):
    st.write("""
    **Key Metrics for Credit Risk:**
    
    - **AUC (Area Under ROC Curve):** Measures separation between defaulters and non-defaulters
      - 0.5 = random | 0.7-0.8 = acceptable | >0.8 = good
    
    - **Gini Coefficient (2*AUC - 1):** Industry standard for model ranking
    
    - **Cross-Validation (CV):** Shows model stability across different data splits
      - Low variance = stable model
      - High variance = overfitting
    
    - **Confusion Matrix Trade-offs:**
      - **False Positives** → Lost revenue (rejecting good clients)
      - **False Negatives** → Loan losses (accepting bad clients)
    
    - **Threshold Selection:** Higher threshold = fewer defaults but lower approval rate
    """)

st.caption("Built with ❤️ by Denys Yakovliev | End-to-End Credit Risk Modeling")