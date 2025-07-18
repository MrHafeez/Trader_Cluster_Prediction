
# Trader Cluster Prediction Dashboard

This is a lightweight Streamlit application for predicting the behavioral cluster of a trader based on key trading metrics. The model uses KMeans clustering over PCA-reduced, scaled features.

## 📌 Project Overview

This application allows users to:
- Input key trade metrics: average quantity, price, PnL, win rate, etc.
- Predict the cluster (behavioral segment) using a pre-trained ML model.
- View the PCA coordinates for further visual analysis.

## 🔧 Components
- `predict_cluster_app.py`: Streamlit UI and backend logic
- `scaler.pkl`, `pca.pkl`, `kmeans_model.pkl`: Saved pre-trained model components
- `.streamlit/config.toml`: Configuration for Streamlit
- `requirements.txt`: Dependency file for setup

## 🧠 Clusters Description (for interpretation)
| Cluster | Description |
|---------|-------------|
| 0       | Low activity, moderate risk |
| 1       | High volume scalper |
| 2       | Infrequent but high-risk |
| 3       | Balanced swing trader |
*Note: Clusters may vary based on your training data.*

## 🚀 Run Locally

1. Clone this repository:
```bash
git clone https://github.com/MrHafeez/trader-clustering-dashboard.git
cd trader-clustering-dashboard
```

2. Create virtual environment and activate:
```bash
python -m venv venv
venv\Scripts\activate    # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run predict_cluster_app.py
```

## 📁 Folder Structure
```
.
├── .streamlit
│   └── config.toml
├── predict_cluster_app.py
├── scaler.pkl
├── pca.pkl
├── kmeans_model.pkl
├── requirements.txt
└── README.md
```

## 📊 PCA Coordinates Output
Useful for visualizing how input points map into the PCA space. You can extend this to plot all clusters using matplotlib or seaborn.

## 🧠 Next Extensions (Optional)
- Visualize clusters in 2D using `plotly` or `seaborn`
- Export prediction to CSV using `st.download_button`
- Add authentication for internal use

## License
This project is MIT licensed. See `LICENSE` for more details.
