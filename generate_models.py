import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

np.random.seed(42)
n = 500

# Cluster 0: Risk-Averse — low qty, low pnl, high win rate, few trades
# Cluster 1: High-Frequency — high trades, low price, moderate pnl
# Cluster 2: Momentum — high price, high pnl volatility, low win rate
# Cluster 3: Balanced — moderate everything

def make_cluster(n, avg_qty, avg_price, avg_pnl, pnl_vol, num_trades, bsr, win_rate, noise=0.1):
    return pd.DataFrame({
        "avg_quantity":   np.random.normal(avg_qty,    avg_qty*noise,    n).clip(1),
        "avg_price":      np.random.normal(avg_price,  avg_price*noise,  n).clip(0),
        "avg_pnl":        np.random.normal(avg_pnl,    abs(avg_pnl)*0.3, n),
        "pnl_volatility": np.random.normal(pnl_vol,    pnl_vol*noise,    n).clip(0),
        "num_trades":     np.random.normal(num_trades, num_trades*noise, n).clip(1),
        "buy_sell_ratio": np.random.normal(bsr,        bsr*noise,        n).clip(0),
        "win_rate":       np.random.normal(win_rate,   0.05,             n).clip(0, 1),
    })

c0 = make_cluster(n, avg_qty=20,  avg_price=500,  avg_pnl=100,  pnl_vol=200,  num_trades=10,  bsr=1.0, win_rate=0.75)
c1 = make_cluster(n, avg_qty=200, avg_price=100,  avg_pnl=50,   pnl_vol=500,  num_trades=200, bsr=1.5, win_rate=0.55)
c2 = make_cluster(n, avg_qty=50,  avg_price=3000, avg_pnl=500,  pnl_vol=3000, num_trades=15,  bsr=2.0, win_rate=0.40)
c3 = make_cluster(n, avg_qty=80,  avg_price=1200, avg_pnl=250,  pnl_vol=1200, num_trades=50,  bsr=1.2, win_rate=0.62)

df = pd.concat([c0, c1, c2, c3], ignore_index=True)

features = ["avg_quantity","avg_price","avg_pnl","pnl_volatility","num_trades","buy_sell_ratio","win_rate"]
X = df[features].values

# Step 1: Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: PCA (reduce to 2 components)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Step 3: KMeans on PCA-reduced data
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_pca)

# Save models
joblib.dump(scaler, "/home/claude/scaler.pkl")
joblib.dump(pca,    "/home/claude/pca.pkl")
joblib.dump(kmeans, "/home/claude/kmeans_model.pkl")

print("✅ scaler.pkl saved")
print("✅ pca.pkl saved")
print("✅ kmeans_model.pkl saved")

# Quick sanity check — simulate predict_cluster_app.py logic
test_input = pd.DataFrame([{
    "avg_quantity": 45.0, "avg_price": 1750.0, "avg_pnl": 200.0,
    "pnl_volatility": 1500.0, "num_trades": 20.0,
    "buy_sell_ratio": 1.5, "win_rate": 0.7
}])

scaled  = scaler.transform(test_input)
reduced = pca.transform(scaled)
cluster = kmeans.predict(reduced)[0]

cluster_names = {0:"Risk-Averse Trader", 1:"High-Frequency Trader", 2:"Momentum Trader", 3:"Balanced Trader"}
print(f"\n🧪 Test prediction → Cluster {cluster}: {cluster_names[cluster]}")
print(f"   PCA coords: {reduced[0]}")
