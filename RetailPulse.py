import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="RetailPulse - AI Retail Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetailPulse - AI Powered Retail Analytics Dashboard")
st.markdown("An end-to-end Data Science & Business Intelligence platform for sales tracking, RFM customer segmentation, time-series demand forecasting, and inventory risk management.")

DATA_PATH = Path("data/raw/online_retail.xlsx")

# -------------------------------------------------
# DATA PREPROCESSING PIPELINE
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(path: Path):
    if not path.exists():
        candidates = list(path.parent.glob("*.xlsx")) + list(path.parent.glob("*.csv"))
        if not candidates:
            return pd.DataFrame()
        path = candidates[0]

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    cust_col = next((cols_lower[c] for c in ["customerid", "customer id", "customer_id"] if c in cols_lower), None)
    inv_col = next((cols_lower[c] for c in ["invoiceno", "invoice no", "invoice_number"] if c in cols_lower), None)
    date_col = next((cols_lower[c] for c in ["invoicedate", "invoice date", "date"] if c in cols_lower), None)
    qty_col = next((cols_lower[c] for c in ["quantity", "qty"] if c in cols_lower), None)
    price_col = next((cols_lower[c] for c in ["unitprice", "unit price", "price"] if c in cols_lower), None)
    desc_col = next((cols_lower[c] for c in ["description", "product", "item"] if c in cols_lower), None)
    country_col = next((cols_lower[c] for c in ["country", "region"] if c in cols_lower), None)

    df["CustomerID"] = df[cust_col].astype(str).str.replace(r"\.0$", "", regex=True) if cust_col else "CUST_UNKNOWN"
    df["InvoiceNo"] = df[inv_col] if inv_col else "INV_UNKNOWN"
    df["InvoiceDate"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["Quantity"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0) if qty_col else 0
    df["UnitPrice"] = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0) if price_col else 0.0
    df["Description"] = df[desc_col] if desc_col else "Item"
    df["Country"] = df[country_col] if country_col else "Unspecified"

    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date
    return df

df = load_data(DATA_PATH)

if df.empty:
    st.error("Data file not found. Please ensure `data/raw/online_retail.xlsx` is present.")
    st.stop()

# Sidebar Controls
st.sidebar.header("Filter Analytics")
countries = sorted(df["Country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect("Countries", countries, default=countries)

filtered_df = df[df["Country"].isin(selected_countries)] if selected_countries else df.copy()

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Customer Segmentation (RFM)", "Demand Forecasting", "Inventory ROP"])

with tab1:
    st.subheader("Sales & Revenue Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${filtered_df['TotalPrice'].sum():,.2f}")
    m2.metric("Total Orders", f"{filtered_df['InvoiceNo'].nunique():,}")
    m3.metric("Total Customers", f"{filtered_df['CustomerID'].nunique():,}")
    m4.metric("Avg Spend / Order", f"${filtered_df['TotalPrice'].sum() / max(1, filtered_df['InvoiceNo'].nunique()):,.2f}")

    daily_sales = filtered_df.groupby("Date")["TotalPrice"].sum().reset_index()
    if PLOTLY_AVAILABLE:
        fig = px.line(daily_sales, x="Date", y="TotalPrice", title="Daily Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(daily_sales.set_index("Date"))

with tab2:
    st.subheader("RFM Customer Clustering")
    snapshot_date = filtered_df["Date"].max() + pd.Timedelta(days=1)
    rfm = filtered_df.groupby("CustomerID").agg(
        Recency=("Date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    if PLOTLY_AVAILABLE:
        fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", color="Cluster", size="Frequency", title="RFM Clusters")
        st.plotly_chart(fig_rfm, use_container_width=True)
    else:
        st.dataframe(rfm.head(50))

with tab3:
    st.subheader("Time Series Revenue Forecast")
    if not PROPHET_AVAILABLE:
        st.warning("Prophet not installed.")
    else:
        daily_prophet = filtered_df.groupby("Date")["TotalPrice"].sum().reset_index()
        daily_prophet.columns = ["ds", "y"]
        daily_prophet["ds"] = pd.to_datetime(daily_prophet["ds"])
        
        m = Prophet()
        m.fit(daily_prophet)
        future = m.make_future_dataframe(periods=30)
        fc = m.predict(future)

        if PLOTLY_AVAILABLE:
            fig_fc = px.line(fc.tail(60), x="ds", y="yhat", title="30-Day Forecast Revenue")
            st.plotly_chart(fig_fc, use_container_width=True)

with tab4:
    st.subheader("Inventory Stock Reorder Planning")
    prod_summary = filtered_df.groupby("Description").agg(
        Units_Sold=("Quantity", "sum"),
        Revenue=("TotalPrice", "sum")
    ).sort_values("Units_Sold", ascending=False).reset_index()
    st.dataframe(prod_summary.head(100), use_container_width=True)
