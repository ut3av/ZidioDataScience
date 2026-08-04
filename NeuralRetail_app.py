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
from sklearn.metrics import silhouette_score

# Set Matplotlib clean enterprise style with eye-pleasing corporate colors
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "grid.color": "#f1f5f9",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})

# -------------------------------------------------
# PAGE CONFIGURATION & CHERRY BLOSSOM STYLING
# -------------------------------------------------
st.set_page_config(
    page_title="NeuralRetail - Executive AI Retail Analytics",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Luminous Theme CSS with Smooth Hover Micro-Animations
st.markdown(
    """
    <style>
    :root {
        --bg-primary: #fff5f7;
        --surface-card: #ffffff;
        --border-color: #fbcfe8;
        --text-main: #1f2937;
        --text-muted: #6b7280;
        --accent-pink: #ec4899;
        --accent-rose: #f43f5e;
        --accent-soft-pink: #fdf2f8;
        --accent-soft-purple: #f3e8ff;
    }
    
    body, .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-main);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Cherry Blossom Landing Hero Banner with Smooth Hover Effects */
    .hero-card {
        background: linear-gradient(135deg, #fff0f5 0%, #fce4ec 45%, #f8bbd0 100%);
        color: #880e4f !important;
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(244, 143, 177, 0.22);
        border: 1px solid #f48fb1;
        position: relative;
        overflow: hidden;
        transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.4s ease;
    }
    .hero-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 40px rgba(244, 143, 177, 0.35);
    }
    
    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.95rem;
        border-radius: 999px;
        background: #ffffff;
        color: #c2185b;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        border: 1px solid #f48fb1;
        box-shadow: 0 2px 8px rgba(194, 24, 91, 0.1);
        transition: transform 0.3s ease, background-color 0.3s ease;
    }
    .hero-badge:hover {
        transform: scale(1.05);
        background-color: #fff0f5;
    }
    
    .hero-title {
        font-size: 2.4rem !important;
        font-weight: 850 !important;
        margin: 0 0 0.5rem 0 !important;
        color: #880e4f !important;
        letter-spacing: -0.02em;
        line-height: 1.25 !important;
    }
    .hero-subtitle {
        color: #ad1457 !important;
        font-size: 1.05rem;
        line-height: 1.6 !important;
        margin: 0;
        font-weight: 500;
    }

    /* Metric Card Hover Animation */
    .metric-card {
        background: var(--surface-card);
        border: 1px solid var(--border-color);
        border-top: 3px solid #f472b6;
        border-radius: 16px;
        padding: 1.25rem 1.1rem;
        box-shadow: 0 4px 15px rgba(244, 114, 182, 0.06);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.015);
        box-shadow: 0 14px 30px rgba(244, 114, 182, 0.2);
        border-color: #ec4899;
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
        border-radius: 12px;
        background: var(--accent-soft-pink);
        font-size: 1.1rem;
        border: 1px solid #fbcfe8;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }
    .metric-card:hover .metric-icon {
        transform: rotate(12deg) scale(1.12);
        background-color: #fbcfe8;
    }
    
    .metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #9d174d;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #831843;
        letter-spacing: -0.02em;
        margin: 0.2rem 0;
        line-height: 1.2;
    }
    .metric-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.4;
    }

    /* Manager Briefing Card Hover Effect */
    .manager-playbook-card {
        background: #fdf2f8;
        border-left: 4px solid #ec4899;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 12px rgba(236, 72, 153, 0.05);
        border: 1px solid #fbcfe8;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-left-width 0.2s ease;
    }
    .manager-playbook-card:hover {
        transform: translateX(6px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.12);
        border-left-width: 6px;
    }
    
    .manager-playbook-title {
        font-weight: 700;
        color: #9d174d;
        font-size: 0.98rem;
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        line-height: 1.3;
    }
    .manager-playbook-text {
        font-size: 0.9rem;
        color: #374151;
        line-height: 1.6;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    /* Tab Button Hover Animation */
    .stTabs [role="tablist"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
        transition: transform 0.2s ease, background-color 0.2s ease !important;
    }
    .stTabs [role="tablist"] button:hover {
        transform: translateY(-2px) !important;
        background-color: #fff0f5 !important;
    }

    .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #fdf2f8 !important;
        color: #be185d !important;
        border: 1px solid #fbcfe8 !important;
    }
    
    /* Button Hover Lift & Glow Animation */
    .stButton>button, .stDownloadButton>button {
        border-radius: 999px !important;
        background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%) !important;
        color: #be185d !important;
        border: 1px solid #fbcfe8 !important;
        font-weight: 700 !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease, background 0.25s ease !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        border-color: #ec4899 !important;
        background: linear-gradient(135deg, #fdf2f8 0%, #fbcfe8 100%) !important;
        box-shadow: 0 8px 20px rgba(236, 72, 153, 0.22) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Landing Page Banner with Preserved User Badge Icon
st.markdown(
    """
    <div class="hero-card">
      <div class="hero-badge">🔭 Retail Intelligence</div>
      <h1 class="hero-title">NeuralRetail Analytics Engine</h1>
      <p class="hero-subtitle">Industry-grade retail analytics platform tailored for Operations & C-Suite Managers: Real-time revenue tracking, RFM customer cohort segmentation, AI-powered 90-day demand forecasting, and automated inventory stockout prevention.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path("data/raw/online_retail.xlsx")

# -------------------------------------------------
# DATA INGESTION & PIPELINE (CACHED)
# -------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Ingesting enterprise retail datasets...")
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
        st.error(f"Error loading data file {target_path}: {e}")
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

    # Data Cleaning & Quality Assurance
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
    st.error(f"Unable to load retail dataset from `{DATA_PATH}`. Please ensure `online_retail.xlsx` is placed in `data/raw/`.")
    st.stop()

# -------------------------------------------------
# SIDEBAR FILTERS WITH MANAGER TOOLTIPS
# -------------------------------------------------
with st.sidebar:
    st.header("🌸 Executive Filters")
    st.caption("Customize the analysis scope across regions, timeframes, and SKU categories.")

    # Country Filter
    all_countries = sorted(raw_df["Country"].dropna().unique().tolist())
    selected_countries = st.multiselect(
        "Geography / Countries",
        all_countries,
        default=all_countries,
        help="ℹ️ Filter metrics by operating territory. Hover over or select specific regions to analyze isolated market performance."
    )

    # Date Range Filter
    min_date = raw_df["Date"].min()
    max_date = raw_df["Date"].max()
    date_range = st.date_input(
        "Analysis Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="ℹ️ Select the start and end dates for historical evaluation. Metrics, segmentation, and forecasts automatically recalculate based on this period."
    )

    # Product Description Filter
    all_products = sorted(raw_df["Description"].dropna().unique().tolist())
    selected_products = st.multiselect(
        "Filter Specific SKUs (Top 500)",
        all_products[:500],
        help="ℹ️ Drill down into individual product lines or merchandise bundles to observe line-item level demand and revenue trends."
    )

    # Text Search Filter
    search_query = st.text_input(
        "SKU Keyword Search",
        placeholder="e.g. HEART, BAG, BOTTLE",
        help="ℹ️ Search product descriptions by keyword (case-insensitive) to filter matching inventory items."
    )

    st.markdown("---")
    st.header("🎨 Display Settings")
    use_plotly = st.checkbox(
        "Enable Interactive Charts (Plotly)",
        value=False,
        help="ℹ️ Toggle ON for interactive WebGL tooltips and zoom. Toggle OFF for server-rendered Matplotlib/Seaborn static images (0% WebGL dependency)."
    )

    st.markdown("---")
    if st.button("Reset All Filters", use_container_width=True, help="ℹ️ Reset all country, date, product, and search filters back to default values."):
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
    st.warning("⚠️ No transactions match your current sidebar filters. Please broaden your country, date, or SKU selection.")
    st.stop()

# -------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------
tab_overview, tab_segment, tab_forecast, tab_inventory, tab_raw = st.tabs([
    "📊 Overview & Executive KPIs",
    "👥 Customer RFM Intelligence",
    "🔮 Predictive Demand Forecasting",
    "📦 Inventory & ROP Optimization",
    "📋 Data Explorer & Audit"
])

# -------------------------------------------------
# TAB 1: OVERVIEW & EXECUTIVE KPIS
# -------------------------------------------------
with tab_overview:
    st.subheader("📊 Business Performance Summary")
    st.caption("Hover over the ℹ️ info icons next to each KPI for manager-level definitions and strategic significance.")

    tot_revenue = filtered_df["TotalPrice"].sum()
    tot_orders = filtered_df["InvoiceNo"].nunique()
    tot_customers = filtered_df["CustomerID"].nunique()
    avg_order_val = tot_revenue / tot_orders if tot_orders > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Gross Revenue 💎",
            value=f"${tot_revenue:,.2f}",
            help="ℹ️ Total monetary revenue generated across all completed orders in the selected scope. Use this metric to track top-line financial growth."
        )
    with c2:
        st.metric(
            label="Total Orders Processed 🛍️",
            value=f"{tot_orders:,}",
            help="ℹ️ Number of unique invoices (transactions) completed. Measures order volume and warehouse operational velocity."
        )
    with c3:
        st.metric(
            label="Active Unique Buyers 🎀",
            value=f"{tot_customers:,}",
            help="ℹ️ Count of distinct registered customer IDs purchasing during this period. Indicates active customer base size."
        )
    with c4:
        st.metric(
            label="Average Order Value (AOV) 💳",
            value=f"${avg_order_val:,.2f}",
            help="ℹ️ Average spend per transaction (Total Revenue ÷ Total Orders). Increasing AOV via upselling and bundling directly boosts profitability."
        )

    # Manager Insight Box
    st.markdown(
        f"""
        <div class="manager-playbook-card">
          <div class="manager-playbook-title">💡 Executive Briefing & Manager Insights</div>
          <div class="manager-playbook-text">
            During the selected period from <b>{start_d}</b> to <b>{end_d}</b>, your business processed <b>{tot_orders:,}</b> orders generating <b>${tot_revenue:,.2f}</b> in revenue. 
            The average spend per basket is <b>${avg_order_val:,.2f}</b> across <b>{tot_customers:,}</b> active buyers.
            Focus marketing strategy on increasing basket size to drive AOV above benchmark targets.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Revenue Trend Line Chart (Corporate Eye-Pleasing Blue/Navy Palette)
    daily_trend = filtered_df.groupby("Date")["TotalPrice"].sum().reset_index()
    daily_trend["Date"] = pd.to_datetime(daily_trend["Date"])
    daily_trend["7D_MA"] = daily_trend["TotalPrice"].rolling(window=7, min_periods=1).mean()

    st.subheader("📈 Daily Sales Trend & 7-Day Moving Average")
    st.caption("ℹ️ The solid line represents daily actual sales; the bold trendline displays the 7-day smoothing average to filter out day-of-week noise.")

    if use_plotly and PLOTLY_AVAILABLE:
        fig_trend = px.line(
            daily_trend, x="Date", y=["TotalPrice", "7D_MA"],
            labels={"value": "Revenue ($)", "variable": "Metric"},
            title="Daily Sales Trend & 7-Day Moving Average",
            color_discrete_map={"TotalPrice": "#60a5fa", "7D_MA": "#2563eb"}
        )
        fig_trend.update_layout(template="plotly_white", height=380, hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 3.8), facecolor="white")
        ax.plot(daily_trend["Date"], daily_trend["TotalPrice"], color="#60a5fa", label="Daily Revenue", alpha=0.7, linewidth=1.5)
        ax.plot(daily_trend["Date"], daily_trend["7D_MA"], color="#2563eb", label="7-Day Moving Average", linewidth=2.5)
        ax.set_ylabel("Revenue ($)", labelpad=8)
        ax.set_xlabel("Date", labelpad=8)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", color="#e2e8f0", alpha=0.7)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close(fig)

    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🏆 Top 10 Products by Revenue")
        st.caption("ℹ️ Identifies top revenue-contributing SKUs. Truncated to avoid text clipping.")
        top_products = filtered_df.groupby("Description")["TotalPrice"].sum().nlargest(10).reset_index()
        top_products["Short_Desc"] = top_products["Description"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig_top = px.bar(
                top_products, x="TotalPrice", y="Short_Desc", orientation="h",
                color="TotalPrice", color_continuous_scale="Blues",
                labels={"TotalPrice": "Revenue ($)", "Short_Desc": "Product"}
            )
            fig_top.update_layout(template="plotly_white", height=360, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
            sns.barplot(data=top_products, x="TotalPrice", y="Short_Desc", palette="Blues_r", ax=ax)
            ax.set_xlabel("Revenue ($)", labelpad=8)
            ax.set_ylabel("")
            ax.grid(True, linestyle="--", color="#e2e8f0", alpha=0.7)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

    with col_chart2:
        st.subheader("🌍 Regional Revenue Distribution")
        st.caption("ℹ️ Horizontal bar ranking eliminates overlapping country labels cleanly.")
        country_rev = filtered_df.groupby("Country")["TotalPrice"].sum().nlargest(8).reset_index()
        country_rev["Share_Pct"] = (country_rev["TotalPrice"] / tot_revenue * 100).round(1)
        country_rev["Label"] = country_rev.apply(lambda r: f"{r['Country']} ({r['Share_Pct']}%)", axis=1)

        if use_plotly and PLOTLY_AVAILABLE:
            fig_country = px.bar(
                country_rev, x="TotalPrice", y="Country", orientation="h",
                color="TotalPrice", color_continuous_scale="Viridis",
                text="Label", labels={"TotalPrice": "Revenue ($)", "Country": "Country"}
            )
            fig_country.update_layout(template="plotly_white", height=360, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
            bars = sns.barplot(data=country_rev, x="TotalPrice", y="Country", palette="crest", ax=ax)
            
            for p in bars.patches:
                width = p.get_width()
                if width > 0:
                    pct = (width / tot_revenue * 100)
                    ax.annotate(f" ${width:,.0f} ({pct:.1f}%)",
                                (width, p.get_y() + p.get_height() / 2.),
                                ha='left', va='center',
                                fontsize=8.5, color='#334155', xytext=(5, 0),
                                textcoords='offset points')

            ax.set_xlabel("Revenue ($)", labelpad=8)
            ax.set_ylabel("")
            ax.set_xlim(0, country_rev["TotalPrice"].max() * 1.35)
            ax.grid(True, linestyle="--", color="#e2e8f0", alpha=0.7)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

# -------------------------------------------------
# TAB 2: RFM CUSTOMER SEGMENTATION
# -------------------------------------------------
with tab_segment:
    st.subheader("👥 Customer RFM Intelligence & Behavioral Clustering")
    st.markdown("Quantifies customer value using **Recency** (days since purchase), **Frequency** (order count), and **Monetary** (total spend).")

    with st.expander("❓ What is RFM Segmentation & How should Managers use it?", expanded=False):
        st.markdown("""
        **RFM Analysis** is an industry-standard framework used by enterprise retailers to tier customer accounts:
        - **Recency (R)**: How recently a customer purchased. Lower recency = higher engagement.
        - **Frequency (F)**: How often they order. High frequency = strong brand loyalty.
        - **Monetary (M)**: Total dollars spent. High monetary = high lifetime value (CLV).

        **Manager Strategic Actions:**
        - **Champions 👑**: Provide VIP perks, early access to launches, and loyalty rewards.
        - **Loyal / Recent 🌟**: Cross-sell premium bundles and encourage subscription referrals.
        - **Potential / Core 🎯**: Offer volume discounts to convert into frequent buyers.
        - **At-Risk / Dormant ⚠️**: Launch automated win-back email campaigns with special discount codes.
        """)

    col_seg_controls, col_seg_blank = st.columns([1.2, 1.8])
    with col_seg_controls:
        n_clusters = st.slider(
            "Select K-Means Cluster Count",
            min_value=3, max_value=6, value=4,
            help="ℹ️ Adjust how granularly the Machine Learning model groups customer accounts into distinct behavioral segments."
        )

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

    # Calculate Data Science Quality Score (Silhouette Metric)
    sil_score = silhouette_score(rfm_scaled, rfm["Cluster"]) if len(rfm) > n_clusters else 0.0

    # Segment Naming Heuristics
    cluster_stats = rfm.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        Count=("CustomerID", "count")
    ).reset_index()

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

    st.caption(f"🤖 **ML Model Health Check**: KMeans Clustering Silhouette Score = **{sil_score:.3f}** (Scores > 0.35 indicate strong segment separation).")

    col_rfm_tbl, col_rfm_chart = st.columns([1.2, 1.8])
    with col_rfm_tbl:
        st.markdown("**Segment Business Summary**")
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
            "Export RFM Segment Target List (CSV)",
            data=rfm.to_csv(index=False),
            file_name="rfm_customer_segments.csv",
            mime="text/csv",
            use_container_width=True,
            help="ℹ️ Download complete list of customer IDs assigned to their respective segments for email marketing campaigns."
        )

    with col_rfm_chart:
        if use_plotly and PLOTLY_AVAILABLE:
            fig_rfm = px.scatter(
                rfm, x="Recency", y="Monetary", color="Segment", size="Frequency",
                hover_data=["CustomerID"], log_y=True,
                color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b", "#ef4444"],
                labels={"Recency": "Recency (Days)", "Monetary": "Total Spend ($) [Log Scale]"},
                title="Customer Segments: Recency vs. Spend"
            )
            fig_rfm.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_rfm, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
            sns.scatterplot(
                data=rfm, x="Recency", y="Monetary", hue="Segment", size="Frequency",
                sizes=(20, 200), alpha=0.8, ax=ax, palette="tab10"
            )
            ax.set_yscale("log")
            ax.set_xlabel("Recency (Days)", labelpad=8)
            ax.set_ylabel("Total Spend ($) [Log Scale]", labelpad=8)
            ax.set_title("Customer Segments: Recency vs Spend")
            ax.grid(True, linestyle="--", color="#e2e8f0", alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

# -------------------------------------------------
# TAB 3: AI DEMAND FORECASTING (PROPHET)
# -------------------------------------------------
with tab_forecast:
    st.subheader("🔮 AI-Powered Time-Series Demand Forecasting")
    st.markdown("Anticipate future revenue and order volume using additive time-series machine learning.")

    with st.expander("❓ How does Demand Forecasting work & how should Managers plan budget?", expanded=False):
        st.markdown("""
        **Facebook Prophet Time-Series Model**:
        - **Historical Baseline**: Analyzes daily sales trends and weekly seasonality (e.g. weekend vs weekday purchasing spikes).
        - **Confidence Intervals (80%)**: The shaded region represents the expected range of outcomes. 
        
        **Manager Planning Guide:**
        - **Upper Bound**: Prepare supplier capacity and staffing for peak sales events.
        - **Lower Bound**: Use for baseline financial cash-flow modeling to guarantee working capital sufficiency.
        """)

    if not PROPHET_AVAILABLE:
        st.error("Prophet library is not available in the runtime environment. Please install `prophet` to enable forecasting.")
    else:
        fc_col1, fc_col2, fc_blank = st.columns([1.2, 1.2, 1.6])
        with fc_col1:
            forecast_metric = st.selectbox(
                "Select Forecast Target",
                ["Revenue ($)", "Order Volume (Invoices)"],
                help="ℹ️ Choose whether to project future dollar revenue or transaction invoice volume."
            )
        with fc_col2:
            forecast_horizon = st.selectbox(
                "Forecast Planning Horizon",
                [7, 14, 30, 60, 90],
                index=2,
                help="ℹ️ Set how many days into the future the AI model should project demand."
            )

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
            with st.spinner(f"Fitting Prophet time-series model for {forecast_horizon}-day horizon..."):
                m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
                m.fit(daily_data)
                
                future = m.make_future_dataframe(periods=forecast_horizon, freq="D")
                forecast_df = m.predict(future)

            hist_plot = daily_data.tail(90)
            future_plot = forecast_df.tail(forecast_horizon)
            
            proj_sum = future_plot["yhat"].clip(lower=0).sum()
            recent_period_sum = hist_plot.tail(forecast_horizon)["y"].sum()
            growth_rate = ((proj_sum - recent_period_sum) / recent_period_sum * 100) if recent_period_sum > 0 else 0.0

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(
                    label=f"Projected {forecast_horizon}-Day Total 🎯",
                    value=f"${proj_sum:,.2f}" if "Revenue" in forecast_metric else f"{proj_sum:,.0f} orders",
                    help=f"ℹ️ Sum of expected daily predictions ({forecast_metric}) over the next {forecast_horizon} days."
                )
            with m_col2:
                st.metric(
                    label="Pacing vs Recent Period 📈",
                    value=f"{growth_rate:+.1f}%",
                    help=f"ℹ️ Percentage change of projected {forecast_horizon}-day total compared to the immediate prior {forecast_horizon}-day historical period."
                )
            with m_col3:
                peak_day = future_plot.loc[future_plot["yhat"].idxmax(), "ds"].strftime("%Y-%m-%d")
                st.metric(
                    label="Expected Peak Demand Date 📅",
                    value=peak_day,
                    help="ℹ️ Specific date within the forecast window predicted to experience maximum customer demand."
                )

            # Visualization (Corporate Royal Blue Data Science Palette)
            if use_plotly and PLOTLY_AVAILABLE:
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=hist_plot["ds"], y=hist_plot["y"],
                    mode="lines", name="Historical Actuals", line=dict(color="#64748b", width=2)
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_plot["ds"], y=future_plot["yhat"].clip(lower=0),
                    mode="lines", name="Forecast Projection", line=dict(color="#2563eb", width=3, dash="dash")
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_plot["ds"].tolist() + future_plot["ds"].tolist()[::-1],
                    y=future_plot["yhat_upper"].clip(lower=0).tolist() + future_plot["yhat_lower"].clip(lower=0).tolist()[::-1],
                    fill="toself", fillcolor="rgba(37, 99, 235, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% Confidence Interval"
                ))
                fig_fc.update_layout(
                    title=f"{forecast_horizon}-Day Demand Forecast Projection",
                    template="plotly_white", height=420, hovermode="x unified",
                    xaxis_title="Date", yaxis_title=forecast_metric
                )
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="white")
                ax.plot(hist_plot["ds"], hist_plot["y"], label="Historical Actuals", color="#64748b", linewidth=2)
                ax.plot(future_plot["ds"], future_plot["yhat"].clip(lower=0), label="Forecast Projection", color="#2563eb", linewidth=2.5, linestyle="--")
                ax.fill_between(
                    future_plot["ds"], future_plot["yhat_lower"].clip(lower=0), future_plot["yhat_upper"].clip(lower=0),
                    color="#2563eb", alpha=0.2, label="80% Confidence Interval"
                )
                ax.set_title(f"{forecast_horizon}-Day Demand Forecast Projection")
                ax.set_xlabel("Date", labelpad=8)
                ax.set_ylabel(forecast_metric, labelpad=8)
                ax.grid(True, linestyle="--", color="#e2e8f0", alpha=0.7)
                ax.legend(loc="upper left")
                plt.tight_layout(pad=1.5)
                st.pyplot(fig)
                plt.close(fig)

            st.download_button(
                "Export Forecast Predictions (CSV)",
                data=forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False),
                file_name=f"demand_forecast_{forecast_horizon}d.csv",
                mime="text/csv",
                help="ℹ️ Download line-item forecast calculations including confidence intervals for inventory supply chain planning."
            )

# -------------------------------------------------
# TAB 4: INVENTORY & REORDER POINT (ROP) PLANNING
# -------------------------------------------------
with tab_inventory:
    st.subheader("📦 Inventory Velocity & Reorder Point (ROP) Optimization")
    st.markdown("Automated calculation of **Safety Stock**, **Lead Time Demand**, and **Stockout Risk Status** to optimize warehouse reordering.")

    with st.expander("❓ How are Reorder Points (ROP) & Safety Stock calculated?", expanded=False):
        st.markdown(r"""
        **Industry Standard Inventory Formula**:
        $$\text{ROP} = (\text{Daily Velocity} \times \text{Lead Time}) + (Z \times \sigma_{\text{daily}} \times \sqrt{\text{Lead Time}})$$
        
        Where:
        - **Daily Velocity**: Average units sold per day over historical span.
        - **Supplier Lead Time**: Days required from issuing purchase order to stock delivery.
        - **Safety Stock**: Buffer stock held to protect against unexpected sales spikes ($Z \times \sigma \times \sqrt{L}$).
        - **Z-Score**: Service level target (95% service level = Z score 1.65).

        **Action Trigger Guidelines:**
        - **CRITICAL REORDER 🔴**: Current stock is below 70% of ROP. Order immediately to prevent stockout!
        - **WARNING ROP 🟡**: Current stock has crossed ROP threshold. Issue standard vendor purchase order.
        - **ADEQUATE 🟢**: Stock is above ROP. No immediate reorder required.
        """)

    inv_c1, inv_c2, inv_blank = st.columns([1.2, 1.2, 1.6])
    with inv_c1:
        supplier_lead_days = st.number_input(
            "Supplier Lead Time (Days)",
            min_value=1, max_value=60, value=7,
            help="ℹ️ Number of calendar days required by your supplier to deliver inventory after a purchase order is placed."
        )
    with inv_c2:
        service_level_z = st.selectbox(
            "Target Service Level",
            ["95% (Z=1.65 - Standard)", "99% (Z=2.33 - Critical SKUs)"],
            index=0,
            help="ℹ️ Desired probability of not stocking out during lead time. 95% is standard retail industry benchmark."
        )
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
    kpi_i1.metric(
        label="Urgent Restock Needed 🚨",
        value=f"{crit_count} SKUs",
        delta=f"-{crit_count}",
        delta_color="inverse",
        help="ℹ️ SKUs whose stock levels have dropped below 70% of their Reorder Point. Action: Issue urgent POs!"
    )
    kpi_i2.metric(
        label="Nearing Reorder Point ⚠️",
        value=f"{warn_count} SKUs",
        help="ℹ️ SKUs currently between 70% and 100% of ROP threshold. Action: Prepare standard vendor reorders."
    )
    kpi_i3.metric(
        label="Healthy Stock Levels ✅",
        value=f"{ok_count} SKUs",
        help="ℹ️ SKUs with adequate inventory buffer exceeding calculated Reorder Points."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Product Inventory Action Plan**")
    
    display_inv = prod_inv.sort_values(by="Total_Units_Sold", ascending=False)[
        ["Description", "Total_Units_Sold", "Daily_Velocity", "Safety_Stock", "Reorder_Point_Units", "Simulated_Current_Stock", "Stock_Status"]
    ].head(100)

    display_inv["Daily_Velocity"] = display_inv["Daily_Velocity"].round(2)
    display_inv["Safety_Stock"] = display_inv["Safety_Stock"].round(1)

    st.dataframe(display_inv, use_container_width=True)

    st.download_button(
        "Download Inventory Reorder Action Plan (CSV)",
        data=prod_inv.to_csv(index=False),
        file_name="inventory_reorder_plan.csv",
        mime="text/csv",
        help="ℹ️ Download complete stock action plan for warehouse managers and purchasing teams."
    )

# -------------------------------------------------
# TAB 5: DATA EXPLORER & AUDIT
# -------------------------------------------------
with tab_raw:
    st.subheader("📋 Dataset Explorer & Operational Audit")
    st.markdown("Inspect, search, and export raw transaction data records.")
    
    col_exp1, col_exp2 = st.columns([2, 1])
    with col_exp1:
        global_search = st.text_input(
            "Global Record Search",
            placeholder="Search by customer ID, invoice number, product description...",
            help="ℹ️ Type any string to instantly filter rows across all data columns."
        )
    
    explorer_df = filtered_df.copy()
    if global_search.strip():
        search_mask = explorer_df.astype(str).apply(lambda row: row.str.contains(global_search.strip(), case=False).any(), axis=1)
        explorer_df = explorer_df[search_mask]

    st.markdown(f"Displaying **{len(explorer_df):,}** matching transaction records.")
    st.dataframe(explorer_df.head(500), use_container_width=True)

    st.download_button(
        "Download Filtered Dataset (CSV)",
        data=explorer_df.to_csv(index=False),
        file_name="cleaned_retail_data.csv",
        mime="text/csv",
        use_container_width=True,
        help="ℹ️ Export filtered transaction records to CSV for offline analysis in Excel or Power BI."
    )
