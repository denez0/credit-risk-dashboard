# Credit Risk Model Dashboard

An interactive dashboard for credit risk classification with EDA, cross-validation, and model comparison.

## 🚀 Features

- **Exploratory Data Analysis** – Target distribution, feature distributions, correlations
- **Model Training** – Logistic Regression & Random Forest
- **Cross-Validation** – 5-fold CV with stability metrics
- **Interactive Controls** – Model selector and probability threshold
- **Business Insights** – ROC, Gini, confusion matrix with trade-off analysis
- **Single Prediction** – Test individual applications

## 🛠️ Technologies

- Python 3.11
- Streamlit – Interactive dashboard
- scikit-learn – ML models & CV
- Pandas/NumPy – Data manipulation
- Matplotlib/Seaborn – Visualizations

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| AUC | Model discrimination power (0.5=random, >0.8=good) |
| Gini | 2*AUC-1, industry standard for credit scoring |
| CV AUC | Cross-validation stability (low variance = robust) |

## 🏃 Quick Start

### Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Docker
```bash
docker-compose up --build
```

Then open: http://localhost:8501

## 📁 Project Structure

```
credit-risk-dashboard/
├── app.py              # Main application
├── Dockerfile          # Container definition
├── docker-compose.yml  # Orchestration
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## 📈 Business Value

- **Risk Assessment** – Predict probability of default
- **Threshold Tuning** – Balance approval rate vs. default risk
- **Feature Insights** – Understand what drives credit risk
- **Model Validation** – Cross-validation ensures reliability

## 👨‍💻 Author

Denys Yakovliev – [GitHub](https://github.com/denez0) | [LinkedIn](https://linkedin.com/in/denys-yakovliev-dev)

## 📝 License

MIT