import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Trader Profiling Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Enhanced dark theme
st.markdown("""
    <style>
    html, body, [class*="st-"] {
        background-color: #0e0e0e;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    .css-1d391kg, .css-1v0mbdj, .css-18e3th9 {
        background-color: #121212 !important;
        border-right: 1px solid #2a2a2a;
    }

    /* Input boxes */
    input, .stNumberInput input, .stSlider div, .stSelectbox, .stTextInput {
        background-color: #1a1a1a !important;
        color: #fff !important;
    }

    .stButton>button {
        background-color: #333333;
        color: white;
        border: 1px solid #555555;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border-radius: 4px;
    }

    .stButton>button:hover {
        background-color: #444444;
        border-color: #888;
    }

    /* Slider values */
    .stSlider .css-1t2d3v1 {
        color: #aaa !important;
    }

    /* Tables */
    .stDataFrame {
        background-color: #1a1a1a !important;
    }

    hr {
        border-color: #333;
    }

    small {
        color: #999;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return scaler, pca, kmeans

scaler, pca, kmeans = load_models()

# Sidebar
with st.sidebar:
    st.header("About the App")
    st.write("Cluster traders into personality types using historical metrics. Powered by KMeans on PCA-reduced data.")
    st.markdown("**Trader Types:**")
    st.markdown("- 0: Risk-Averse Trader")
    st.markdown("- 1: High-Frequency Trader")
    st.markdown("- 2: Momentum Trader")
    st.markdown("- 3: Balanced Trader")

# Main content
st.title("Clustering")
st.caption("Use trading behavior to predict trader personality type.")

cluster_names = {
    0: "Risk-Averse Trader",
    1: "High-Frequency Trader",
    2: "Momentum Trader",
    3: "Balanced Trader"
}

with st.form("input_form"):
    st.subheader("Enter Trading Metrics")

    avg_quantity = st.number_input("Average Quantity", min_value=1.0, max_value=100000.0, value=45.0)
    st.markdown("<small>Min: 1.0 &nbsp;&nbsp; Max: 100000.0</small>", unsafe_allow_html=True)

    avg_price = st.number_input("Average Price", min_value=0.0, max_value=100000.0, value=1750.0)
    st.markdown("<small>Min: 0.0 &nbsp;&nbsp; Max: 100000.0</small>", unsafe_allow_html=True)

    avg_pnl = st.number_input("Average PnL", value=200.0)
    st.markdown("<small>No strict range (depends on strategy)</small>", unsafe_allow_html=True)

    pnl_volatility = st.number_input("PnL Volatility", value=1500.0)
    st.markdown("<small>No strict range</small>", unsafe_allow_html=True)

    num_trades = st.number_input("Number of Trades", min_value=1.0, max_value=10000.0, value=20.0)
    st.markdown("<small>Min: 1.0 &nbsp;&nbsp; Max: 10000.0</small>", unsafe_allow_html=True)

    buy_sell_ratio = st.number_input("Buy/Sell Ratio", value=1.5)
    st.markdown("<small>Recommended range: 0.0 – 10.0</small>", unsafe_allow_html=True)

    st.markdown("**Win Rate**")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.markdown("0.0")
    with col2:
        win_rate = st.slider("", min_value=0.0, max_value=1.0, value=0.7)
    with col3:
        st.markdown("1.0")

    submit = st.form_submit_button("Predict Cluster")

# Predict
if submit:
    input_df = pd.DataFrame([{
        "avg_quantity": avg_quantity,
        "avg_price": avg_price,
        "avg_pnl": avg_pnl,
        "pnl_volatility": pnl_volatility,
        "num_trades": num_trades,
        "buy_sell_ratio": buy_sell_ratio,
        "win_rate": win_rate
    }])

    scaled = scaler.transform(input_df)
    reduced = pca.transform(scaled)
    cluster = kmeans.predict(scaled)[0]

    st.success(f"Predicted Cluster: {cluster} — {cluster_names.get(cluster, 'Unknown')}")

    coords = pd.DataFrame(reduced, columns=["PCA 1", "PCA 2"])
    st.write("PCA Coordinates:")
    st.dataframe(coords)

    st.altair_chart(
        alt.Chart(coords.reset_index()).mark_circle(size=100).encode(
            x='PCA 1', y='PCA 2', tooltip=['PCA 1', 'PCA 2']
        ).properties(width=500, height=300).interactive(),
        use_container_width=True
    )

    csv = coords.to_csv(index=False).encode()
    st.download_button("Download PCA Coordinates", data=csv, file_name="pca_coords.csv", mime="text/csv")

# Footer
st.markdown(
    "<hr style='margin-top: 40px;'>"
    "<div style='text-align: center; font-size: 12px;'>"
    "Developed by MrHafeez | Trader Profiling ML App | 2025"
    "</div>",
    unsafe_allow_html=True
)
