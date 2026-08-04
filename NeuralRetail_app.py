import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

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
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------
st.set_page_config(
    page_title="NeuralRetail - AI Retail Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern CSS Theme
st.markdown(
    """
    <style>
    :root {
        --bg-primary: #0f172a;
        --surface-card: #ffffff;
        --border-color: #e2e8f0;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --accent-blue: #2563eb;
        --accent-soft: #eff6ff;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #ef4444;
    }
    
    body, .stApp {
        background-color: #f8fafc;
        color: var(--text-main);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff !important;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    .hero-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.25);
        color: #60a5fa;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    .hero-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.4rem 0 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8 !important;
        font-size: 1.02rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .metric-card {
        background: var(--surface-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.25rem 1.1rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .metric-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-icon {
        display: inline-flex;
        width: 2.2rem;
        height: 2.2rem;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: var(--accent-soft);
        font-size: 1.1rem;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text-main);
        letter-spacing: -0.02em;
        margin: 0.2rem 0;
    }
    .metric-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    
    .status-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .status-urgent { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .status-warning { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
    .status-healthy { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
    
    div[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    .stTabs [role="tablist"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
    <div class="hero-card">
      <div class="hero-badge">Enterprise Intelligence Platform</div>
      <h1 class="hero-title">NeuralRetail Analytics Engine</h1>
      <p class="hero-subtitle">Real-time revenue monitoring, RFM customer segmentation, AI-driven demand forecasting, and automated stockout risk prevention.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path("data/raw/online_retail.xlsx")

# -------------------------------------------------
# DATA INGESTION & PIPELINE (CACHED)
# -------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading and cleaning retail dataset...")
def load_and_preprocess_data(path: Path):
    target_path = path
    if not target_path.exists():
        folder = target_path.parent
        candidates = list(folder.glob("*.xlsx")) + list(folder.glob("*.csv"))
        if not candidates:
            return pd.DataFrame()
        target_path = candidates[0]

    try:
        if target_path.suffix.lower() == ".csv":
            df = pd.read_csv(target_path)
        else:
            df = pd.read_excel(target_path)
    except Exception as e:
        st.error(f"Error loading file {target_path}: {e}")
        return pd.DataFrame()

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Dynamic Column Mapping
    cust_col = next((cols_lower[c] for c in ["customerid", "customer id", "customer_id", "custid"] if c in cols_lower), None)
    inv_col = next((cols_lower[c] for c in ["invoiceno", "invoice no", "invoice_number", "orderid"] if c in cols_lower), None)
    date_col = next((cols_lower[c] for c in ["invoicedate", "invoice date", "date", "transactiondate"] if c in cols_lower), None)
    qty_col = next((cols_lower[c] for c in ["quantity", "qty"] if c in cols_lower), None)
    price_col = next((cols_lower[c] for c in ["unitprice", "unit price", "price"] if c in cols_lower), None)
    desc_col = next((cols_lower[c] for c in ["description", "product", "item"] if c in cols_lower), None)
    country_col = next((cols_lower[c] for c in ["country", "region", "location"] if c in cols_lower), None)

    if cust_col:
        df = df.rename(columns={cust_col: "CustomerID"})
    else:
        df["CustomerID"] = np.nan

    if inv_col:
        df = df.rename(columns={inv_col: "InvoiceNo"})
    else:
        df["InvoiceNo"] = [f"INV_{i}" for i in range(len(df))]

    if date_col:
        df["InvoiceDate"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["InvoiceDate"] = pd.NaT

    if qty_col:
        df = df.rename(columns={qty_col: "Quantity"})
    else:
        df["Quantity"] = 0

    if price_col:
        df = df.rename(columns={price_col: "UnitPrice"})
    else:
        df["UnitPrice"] = 0.0

    if desc_col:
        df = df.rename(columns={desc_col: "Description"})
    else:
        df["Description"] = "Unknown Item"

    if country_col:
        df = df.rename(columns={country_col: "Country"})
    else:
        df["Country"] = "Unspecified"

    # Data Quality Filters
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date

    return df

raw_df = load_and_preprocess_data(DATA_PATH)

if raw_df.empty:
    st.error(f"Unable to load retail dataset from `{DATA_PATH}`. Please place `online_retail.xlsx` in `data/raw/`.")
    st.stop()

# -------------------------------------------------
# SIDEBAR FILTERS & CONTROLS
# -------------------------------------------------
with st.sidebar:
    st.header("🔍 Dataset Filters")
    st.caption("Refine analytics view by geography, dates, and products.")

    # Country Filter
    all_countries = sorted(raw_df["Country"].dropna().unique().tolist())
    selected_countries = st.multiselect("Select Countries", all_countries, default=all_countries)

    # Date Range Filter
    min_date = raw_df["Date"].min()
    max_date = raw_df["Date"].max()
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # Product Description Filter
    all_products = sorted(raw_df["Description"].dropna().unique().tolist())
    selected_products = st.multiselect("Filter Products (Top 500)", all_products[:500])

    # Text Search Filter
    search_query = st.text_input("Product Search", placeholder="e.g. HEART, BAG, BOTTLE")

    st.markdown("---")
    if st.button("Reset All Filters", use_container_width=True):
        st.rerun()

# Apply Filters
filtered_df = raw_df.copy()

if selected_countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[(filtered_df["Date"] >= start_d) & (filtered_df["Date"] <= end_d)]

if selected_products:
    filtered_df = filtered_df[filtered_df["Description"].isin(selected_products)]

if search_query.strip():
    filtered_df = filtered_df[filtered_df["Description"].str.contains(search_query.strip(), case=False, na=False)]

# Guard check for empty filtered dataframe
if filtered_df.empty:
    st.warning("⚠️ No transactions match your current sidebar filters. Please adjust your selection.")
    st.stop()

# -------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------
tab_overview, tab_segment, tab_forecast, tab_inventory, tab_raw = st.tabs([
    "📊 Overview & KPIs",
    "👥 RFM Segmentation",
    "🔮 AI Demand Forecast",
    "📦 Inventory & ROP",
    "📋 Data Explorer"
])

# -------------------------------------------------
# TAB 1: OVERVIEW & KPIS
# -------------------------------------------------
with tab_overview:
    tot_revenue = filtered_df["TotalPrice"].sum()
    tot_orders = filtered_df["InvoiceNo"].nunique()
    tot_customers = filtered_df["CustomerID"].nunique()
    avg_order_val = tot_revenue / tot_orders if tot_orders > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
              <div class="metric-header"><span class="metric-icon">💰</span><span class="metric-title">Total Revenue</span></div>
              <div class="metric-value">${tot_revenue:,.2f}</div>
              <div class="metric-subtitle">Across selected scope</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
              <div class="metric-header"><span class="metric-icon">📦</span><span class="metric-title">Total Invoices</span></div>
              <div class="metric-value">{tot_orders:,}</div>
              <div class="metric-subtitle">Completed orders</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card">
              <div class="metric-header"><span class="metric-icon">👥</span><span class="metric-title">Active Customers</span></div>
              <div class="metric-value">{tot_customers:,}</div>
              <div class="metric-subtitle">Unique buyers</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="metric-card">
              <div class="metric-header"><span class="metric-icon">💳</span><span class="metric-title">Avg Order Value</span></div>
              <div class="metric-value">${avg_order_val:,.2f}</div>
              <div class="metric-subtitle">Revenue per invoice</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Revenue Trend Line Chart
    daily_trend = filtered_df.groupby("Date")["TotalPrice"].sum().reset_index()
    daily_trend["Date"] = pd.to_datetime(daily_trend["Date"])
    daily_trend["7D_MA"] = daily_trend["TotalPrice"].rolling(window=7, min_periods=1).mean()

    if PLOTLY_AVAILABLE:
        fig_trend = px.line(
            daily_trend, x="Date", y=["TotalPrice", "7D_MA"],
            labels={"value": "Revenue ($)", "variable": "Metric"},
            title="Daily Sales Trend & 7-Day Moving Average",
            color_discrete_map={"TotalPrice": "#93c5fd", "7D_MA": "#2563eb"}
        )
        fig_trend.update_layout(
            template="plotly_white",
            height=380,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.line_chart(daily_trend.set_index("Date")[["TotalPrice", "7D_MA"]])

    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🏆 Top 10 Products by Revenue")
        top_products = filtered_df.groupby("Description")["TotalPrice"].sum().nlargest(10).reset_index()
        if PLOTLY_AVAILABLE:
            fig_top = px.bar(
                top_products, x="TotalPrice", y="Description", orientation="h",
                color="TotalPrice", color_continuous_scale="Blues",
                labels={"TotalPrice": "Revenue ($)", "Description": "Product"}
            )
            fig_top.update_layout(template="plotly_white", height=360, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.dataframe(top_products)

    with col_chart2:
        st.subheader("🌍 Revenue Distribution by Country")
        country_rev = filtered_df.groupby("Country")["TotalPrice"].sum().nlargest(8).reset_index()
        if PLOTLY_AVAILABLE:
            fig_country = px.pie(
                country_rev, values="TotalPrice", names="Country", hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_country.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.dataframe(country_rev)

# -------------------------------------------------
# TAB 2: RFM CUSTOMER SEGMENTATION
# -------------------------------------------------
with tab_segment:
    st.subheader("👥 Customer RFM Analytics & Clustering")
    st.markdown("Segment buyers using **Recency** (days since last purchase), **Frequency** (order count), and **Monetary** (total spend).")

    col_seg_controls, col_seg_blank = st.columns([1, 2])
    with col_seg_controls:
        n_clusters = st.slider("Select Target Clusters (K-Means)", min_value=3, max_value=6, value=4)

    snapshot_date = filtered_df["Date"].max() + pd.Timedelta(days=1)
    
    rfm = filtered_df.groupby("CustomerID").agg(
        Recency=("Date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    # Log transformation for skewed distributions & standard scaling
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Segment Naming Heuristics
    cluster_stats = rfm.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        Count=("CustomerID", "count")
    ).reset_index()

    # Assign business labels dynamically based on rank
    cluster_stats["R_Rank"] = cluster_stats["Recency_mean"].rank(ascending=True)
    cluster_stats["M_Rank"] = cluster_stats["Monetary_mean"].rank(ascending=False)
    
    segment_names = []
    for idx, row in cluster_stats.iterrows():
        if row["M_Rank"] == 1:
            segment_names.append("Champions 👑")
        elif row["R_Rank"] == 1:
            segment_names.append("Loyal / Recent 🌟")
        elif row["R_Rank"] == cluster_stats["R_Rank"].max():
            segment_names.append("At-Risk / Dormant ⚠️")
        else:
            segment_names.append("Potential / Core 🎯")

    cluster_stats["Segment"] = segment_names
    label_dict = dict(zip(cluster_stats["Cluster"], cluster_stats["Segment"]))
    rfm["Segment"] = rfm["Cluster"].map(label_dict)

    col_rfm_tbl, col_rfm_chart = st.columns([1.2, 1.8])
    with col_rfm_tbl:
        st.markdown("**Segment Characteristics**")
        summary_display = rfm.groupby("Segment").agg(
            Buyers=("CustomerID", "count"),
            Avg_Recency_Days=("Recency", "mean"),
            Avg_Orders=("Frequency", "mean"),
            Avg_Spend=("Monetary", "mean")
        ).reset_index()
        summary_display["Avg_Recency_Days"] = summary_display["Avg_Recency_Days"].round(1)
        summary_display["Avg_Orders"] = summary_display["Avg_Orders"].round(1)
        summary_display["Avg_Spend"] = summary_display["Avg_Spend"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(summary_display, use_container_width=True)

        st.download_button(
            "Export RFM Segment Data (CSV)",
            data=rfm.to_csv(index=False),
            file_name="rfm_customer_segments.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_rfm_chart:
        if PLOTLY_AVAILABLE:
            fig_rfm = px.scatter(
                rfm, x="Recency", y="Monetary", color="Segment", size="Frequency",
                hover_data=["CustomerID"], log_y=True,
                labels={"Recency": "Recency (Days)", "Monetary": "Total Spend ($) [Log Scale]"},
                title="Customer Segments: Recency vs. Spend"
            )
            fig_rfm.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_rfm, use_container_width=True)
        else:
            st.dataframe(rfm.head(50))

# -------------------------------------------------
# TAB 3: AI DEMAND FORECASTING (PROPHET)
# -------------------------------------------------
with tab_forecast:
    st.subheader("🔮 AI-Powered Time-Series Demand Forecasting")
    st.markdown("Anticipate future revenue and order volume using additive time-series decomposition.")

    if not PROPHET_AVAILABLE:
        st.error("Prophet library is not available in the runtime environment. Please install `prophet` to enable forecasting.")
    else:
        fc_col1, fc_col2, fc_blank = st.columns([1, 1, 2])
        with fc_col1:
            forecast_metric = st.selectbox("Forecast Metric", ["Revenue ($)", "Order Volume (Invoices)"])
        with fc_col2:
            forecast_horizon = st.selectbox("Forecast Horizon", [7, 14, 30, 60, 90], index=2)

        # Prepare continuous daily aggregate
        if forecast_metric == "Revenue ($)":
            daily_data = filtered_df.groupby("Date")["TotalPrice"].sum().reset_index()
            daily_data.columns = ["ds", "y"]
        else:
            daily_data = filtered_df.groupby("Date")["InvoiceNo"].nunique().reset_index()
            daily_data.columns = ["ds", "y"]

        daily_data["ds"] = pd.to_datetime(daily_data["ds"])
        
        # Ensure continuous daily series with zero-filling
        full_date_range = pd.date_range(start=daily_data["ds"].min(), end=daily_data["ds"].max(), freq="D")
        daily_data = daily_data.set_index("ds").reindex(full_date_range, fill_value=0).reset_index()
        daily_data.columns = ["ds", "y"]

        if len(daily_data) < 14:
            st.warning("Insufficient continuous historical data points (less than 14 days) to generate a reliable forecast.")
        else:
            with st.spinner(f"Fitting Prophet model for {forecast_horizon}-day forecast..."):
                m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
                m.fit(daily_data)
                
                future = m.make_future_dataframe(periods=forecast_horizon, freq="D")
                forecast_df = m.predict(future)

            # Split historical vs predicted
            hist_plot = daily_data.tail(90)
            future_plot = forecast_df.tail(forecast_horizon)
            
            proj_sum = future_plot["yhat"].clip(lower=0).sum()
            recent_period_sum = hist_plot.tail(forecast_horizon)["y"].sum()
            growth_rate = ((proj_sum - recent_period_sum) / recent_period_sum * 100) if recent_period_sum > 0 else 0.0

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(f"Projected {forecast_horizon}-Day Total", f"${proj_sum:,.2f}" if "Revenue" in forecast_metric else f"{proj_sum:,.0f} units")
            with m_col2:
                st.metric("Projected Growth vs Recent", f"{growth_rate:+.1f}%")
            with m_col3:
                peak_day = future_plot.loc[future_plot["yhat"].idxmax(), "ds"].strftime("%Y-%m-%d")
                st.metric("Expected Peak Demand Date", peak_day)

            # Combined Visualization
            if PLOTLY_AVAILABLE:
                fig_fc = go.Figure()
                
                # Historical Line
                fig_fc.add_trace(go.Scatter(
                    x=hist_plot["ds"], y=hist_plot["y"],
                    mode="lines", name="Historical Actuals", line=dict(color="#64748b", width=2)
                ))
                
                # Forecast Line
                fig_fc.add_trace(go.Scatter(
                    x=future_plot["ds"], y=future_plot["yhat"].clip(lower=0),
                    mode="lines", name="Forecast Projection", line=dict(color="#2563eb", width=3, dash="dash")
                ))

                # Confidence Intervals
                fig_fc.add_trace(go.Scatter(
                    x=future_plot["ds"].tolist() + future_plot["ds"].tolist()[::-1],
                    y=future_plot["yhat_upper"].clip(lower=0).tolist() + future_plot["yhat_lower"].clip(lower=0).tolist()[::-1],
                    fill="toself", fillcolor="rgba(37, 99, 235, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% Confidence Interval"
                ))

                fig_fc.update_layout(
                    title=f"{forecast_horizon}-Day Demand Forecast Projection",
                    template="plotly_white",
                    height=420,
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title=forecast_metric
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            st.download_button(
                "Download Forecast Data (CSV)",
                data=forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False),
                file_name=f"demand_forecast_{forecast_horizon}d.csv",
                mime="text/csv"
            )

# -------------------------------------------------
# TAB 4: INVENTORY & REORDER POINT (ROP) PLANNING
# -------------------------------------------------
with tab_inventory:
    st.subheader("📦 Inventory Velocity & Reorder Point (ROP) Analysis")
    st.markdown("Automated calculation of **Safety Stock**, **Lead Time Demand**, and **Stockout Risk Status** based on historical consumption velocity.")

    inv_c1, inv_c2, inv_blank = st.columns([1, 1, 2])
    with inv_c1:
        supplier_lead_days = st.number_input("Supplier Lead Time (Days)", min_value=1, max_value=60, value=7)
    with inv_c2:
        service_level_z = st.selectbox("Desired Service Level", ["95% (Z=1.65)", "99% (Z=2.33)"], index=0)
        z_score = 1.65 if "95%" in service_level_z else 2.33

    # Calculate product velocity
    date_span_days = max(1, (filtered_df["Date"].max() - filtered_df["Date"].min()).days + 1)
    
    prod_inv = filtered_df.groupby("Description").agg(
        Total_Units_Sold=("Quantity", "sum"),
        Total_Revenue=("TotalPrice", "sum"),
        Sales_Days=("Date", "nunique"),
        Daily_Std_Dev=("Quantity", "std")
    ).reset_index()

    prod_inv["Daily_Velocity"] = prod_inv["Total_Units_Sold"] / date_span_days
    prod_inv["Daily_Std_Dev"] = prod_inv["Daily_Std_Dev"].fillna(0.0)

    # ROP = (Average Daily Demand * Lead Time) + (Z * StdDev * sqrt(Lead Time))
    prod_inv["Lead_Time_Demand"] = prod_inv["Daily_Velocity"] * supplier_lead_days
    prod_inv["Safety_Stock"] = z_score * prod_inv["Daily_Std_Dev"] * np.sqrt(supplier_lead_days)
    prod_inv["Reorder_Point_Units"] = np.ceil(prod_inv["Lead_Time_Demand"] + prod_inv["Safety_Stock"]).astype(int)

    # Simulated Current Stock Level for demonstration
    prod_inv["Simulated_Current_Stock"] = np.ceil(prod_inv["Reorder_Point_Units"] * np.random.default_rng(42).uniform(0.4, 1.6, len(prod_inv))).astype(int)

    def classify_risk(row):
        if row["Simulated_Current_Stock"] <= row["Reorder_Point_Units"] * 0.7:
            return "CRITICAL REORDER 🔴"
        elif row["Simulated_Current_Stock"] <= row["Reorder_Point_Units"]:
            return "WARNING ROP 🟡"
        else:
            return "ADEQUATE 🟢"

    prod_inv["Stock_Status"] = prod_inv.apply(classify_risk, axis=1)

    crit_count = (prod_inv["Stock_Status"] == "CRITICAL REORDER 🔴").sum()
    warn_count = (prod_inv["Stock_Status"] == "WARNING ROP 🟡").sum()
    ok_count = (prod_inv["Stock_Status"] == "ADEQUATE 🟢").sum()

    kpi_i1, kpi_i2, kpi_i3 = st.columns(3)
    kpi_i1.metric("Urgent Restock Needed", f"{crit_count} SKUs", delta=f"-{crit_count}", delta_color="inverse")
    kpi_i2.metric("Nearing Reorder Point", f"{warn_count} SKUs")
    kpi_i3.metric("Healthy Stock Levels", f"{ok_count} SKUs")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Product Inventory Action Plan**")
    
    display_inv = prod_inv.sort_values(by="Total_Units_Sold", ascending=False)[
        ["Description", "Total_Units_Sold", "Daily_Velocity", "Safety_Stock", "Reorder_Point_Units", "Simulated_Current_Stock", "Stock_Status"]
    ].head(100)

    display_inv["Daily_Velocity"] = display_inv["Daily_Velocity"].round(2)
    display_inv["Safety_Stock"] = display_inv["Safety_Stock"].round(1)

    st.dataframe(display_inv, use_container_width=True)

    st.download_button(
        "Download Inventory ROP Plan (CSV)",
        data=prod_inv.to_csv(index=False),
        file_name="inventory_reorder_plan.csv",
        mime="text/csv"
    )

# -------------------------------------------------
# TAB 5: DATA EXPLORER & RAW DATA
# -------------------------------------------------
with tab_raw:
    st.subheader("📋 Dataset Explorer")
    
    col_exp1, col_exp2 = st.columns([2, 1])
    with col_exp1:
        global_search = st.text_input("Filter Raw Records", placeholder="Search by customer, description, invoice...")
    
    explorer_df = filtered_df.copy()
    if global_search.strip():
        search_mask = explorer_df.astype(str).apply(lambda row: row.str.contains(global_search.strip(), case=False).any(), axis=1)
        explorer_df = explorer_df[search_mask]

    st.markdown(f"Displaying **{len(explorer_df):,}** matching rows.")
    st.dataframe(explorer_df.head(500), use_container_width=True)

    st.download_button(
        "Download Cleaned Dataset (CSV)",
        data=explorer_df.to_csv(index=False),
        file_name="cleaned_retail_data.csv",
        mime="text/csv",
        use_container_width=True
    )
