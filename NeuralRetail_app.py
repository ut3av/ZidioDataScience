import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from pathlib import Path
import json
import os

DATA_PATH = Path("data/raw/online_retail.xlsx")

st.set_page_config(page_title="NeuralRetail Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    "<style>"
    ":root { --bg: #ffffff; --surface: #f8fbff; --border: #e5e7eb; --text: #111827; --muted: #6b7280; --accent: #2563eb; --accent-soft: #eff6ff; }"
    "body, .block-container, .main, .stApp, section[data-testid=stSidebar], [data-testid=stAppViewContainer], [data-testid=stHeader], [data-testid=stToolbar] { background-color: var(--bg) !important; color: var(--text) !important; }"
    "div[data-testid='stSidebar'] { background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important; border-right: 1px solid var(--border) !important; }"
    "h1, h2, h3, h4, p, li, .st-emotion-cache-1wmy9hl { color: var(--text) !important; }"
    ".hero-card { background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); border: 1px solid var(--border); border-radius: 20px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(37, 99, 235, 0.05); }"
    ".hero-badge { display:inline-block; padding:0.35rem 0.7rem; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-size:0.8rem; font-weight:700; letter-spacing:0.02em; margin-bottom:0.6rem; }"
    ".hero-title { font-size:2rem !important; font-weight:800 !important; margin:0 0 .25rem 0 !important; color:#0f172a !important; }"
    ".hero-subtitle { color:var(--muted) !important; font-size:0.98rem; line-height:1.6; margin:0; }"
    ".metric-card { background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1rem 0.85rem; min-height: 145px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04); }"
    ".metric-icon { display:inline-flex; width:2.1rem; height:2.1rem; align-items:center; justify-content:center; border-radius:999px; background:var(--accent-soft); font-size:1rem; margin-bottom:0.6rem; }"
    ".metric-title { font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted); margin-bottom:0.25rem; }"
    ".metric-value { font-size:1.35rem; font-weight:800; color:#111827; margin-bottom:0.2rem; }"
    ".metric-subtitle { font-size:0.9rem; color:var(--muted); }"
    ".filter-card, .panel-card { background: #ffffff; border: 1px solid var(--border); border-radius: 14px; padding: .9rem .95rem; margin-bottom: .8rem; box-shadow: 0 6px 18px rgba(15,23,42,0.04); }"
    ".filter-title { font-weight:700; font-size:0.98rem; margin-bottom:0.25rem; color:#111827; }"
    ".filter-subtitle { font-size:0.85rem; color:var(--muted); line-height:1.45; }"
    ".stButton>button, .stDownloadButton>button { border-radius: 999px !important; border: 1px solid #dbeafe !important; background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%) !important; color: var(--accent) !important; font-weight: 700 !important; padding: 0.45rem 0.95rem !important; }"
    ".stButton>button:hover, .stDownloadButton>button:hover { border-color: var(--accent) !important; box-shadow: 0 6px 16px rgba(37, 99, 235, 0.12) !important; }"
    ".stTabs [role='tablist'] button { border-radius: 999px !important; color: #4b5563 !important; border: 1px solid transparent !important; padding: 0.35rem 0.8rem !important; }"
    ".stTabs [role='tablist'] button[aria-selected='true'] { background: var(--accent-soft) !important; color: var(--accent) !important; border-color: #bfdbfe !important; }"
    "div[data-testid='stDataFrame'] { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }"
    "div[data-testid='stDataFrame'] table thead th, div[data-testid='stDataFrame'] table tbody td { border-color: var(--border) !important; color: var(--text) !important; }"
    "div[data-testid='stDataFrame'] table thead th { background-color: #f9fafb !important; font-weight: 700 !important; }"
    ".block-container { padding-top: 1.3rem !important; }"
    ".stAlert, .stInfo, .stSuccess, .stWarning { border-radius: 14px !important; border: 1px solid var(--border) !important; }"
    ".stMultiSelect div, .stSelectbox div, .stTextInput div, .stDateInput div { border-radius: 10px !important; }"
    "hr { border-color: #e5e7eb !important; }"
    "</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
      <div class="hero-badge">AI Retail Analytics</div>
      <h1 class="hero-title">NeuralRetail Dashboard</h1>
      <p class="hero-subtitle">Track revenue performance, identify valuable customer segments, anticipate demand, and support smarter inventory decisions in a more focused, polished view.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(path: Path):
    # load file (excel or csv)
    if not path.exists():
        # try to find any CSV/Excel in data/raw
        folder = path.parent
        candidates = list(folder.glob('*.csv')) + list(folder.glob('*.xlsx'))
        if not candidates:
            return None
        path = candidates[0]

    if path.suffix.lower() in ('.csv',):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # common column detection
    cols_lower = {c.lower(): c for c in df.columns}

    # detect customer id column
    customer_candidates = ['customerid', 'customer id', 'customer_id', 'customer', 'custid', 'cust_id']
    customer_col = None
    for cand in customer_candidates:
        if cand in cols_lower:
            customer_col = cols_lower[cand]
            break

    if customer_col is None:
        st.error('No customer identifier column found in data. Available columns:')
        st.write(list(df.columns))
        return pd.DataFrame()

    # standardize column name
    df = df.rename(columns={customer_col: 'CustomerID'})

    # detect invoice/date/quantity/unitprice/description columns
    invoice_candidates = ['invoiceno', 'invoice no', 'invoice_number', 'invoice', 'orderid', 'order id', 'order_no', 'order']
    date_candidates = ['invoicedate', 'invoice date', 'date', 'transactiondate']
    qty_candidates = ['quantity', 'qty']
    price_candidates = ['unitprice', 'unit price', 'price', 'unit_price']
    desc_candidates = ['description', 'product', 'item']
    invoicen_col = None

    # invoice column
    invoice_col = None
    for cand in invoice_candidates:
        if cand in cols_lower:
            invoice_col = cols_lower[cand]
            break

    for cand in date_candidates:
        if cand in cols_lower:
            date_col = cols_lower[cand]
            break
    else:
        date_col = None

    for cand in qty_candidates:
        if cand in cols_lower:
            qty_col = cols_lower[cand]
            break
    else:
        qty_col = None

    for cand in price_candidates:
        if cand in cols_lower:
            price_col = cols_lower[cand]
            break
    else:
        price_col = None

    for cand in desc_candidates:
        if cand in cols_lower:
            desc_col = cols_lower[cand]
            break
    else:
        desc_col = None

    # apply basic filters if columns exist
    if qty_col is not None:
        df = df[df[qty_col] > 0]
        df = df.rename(columns={qty_col: 'Quantity'})

    if price_col is not None:
        df = df[df[price_col] > 0]
        df = df.rename(columns={price_col: 'UnitPrice'})

    if date_col is not None:
        df['InvoiceDate'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        # try common date-like column
        if 'date' in cols_lower:
            df['InvoiceDate'] = pd.to_datetime(df[cols_lower['date']], errors='coerce')

    if desc_col is not None:
        df = df.rename(columns={desc_col: 'Description'})

    if invoice_col is not None:
        df = df.rename(columns={invoice_col: 'InvoiceNo'})


    # compute TotalPrice
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # ensure CustomerID exists
    if 'CustomerID' not in df.columns:
        return pd.DataFrame()

    return df


df = load_data(DATA_PATH)
if df is None or df.empty:
    st.error(f"Raw data file not found or no valid rows: {DATA_PATH.resolve()}")
    st.write("Please place the raw Excel file at `data/raw/online_retail.xlsx` relative to this notebook.")
    st.stop()

# --------- UI: controls and layout ---------
# initialize session filters
if 'filters' not in st.session_state:
    st.session_state['filters'] = {}
if 'prefs' not in st.session_state:
    st.session_state['prefs'] = {}

# persistence file for simple local preferences
PREFS_FILE = Path('.streamlit_prefs.json')

def load_prefs():
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                st.session_state['prefs'] = json.load(f)
        except Exception:
            st.session_state['prefs'] = {}

def save_prefs():
    try:
        with open(PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.get('prefs', {}), f)
    except Exception:
        pass

load_prefs()

def set_pref(key, value):
    st.session_state['prefs'][key] = value
    save_prefs()

def safe_rerun():
    try:
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
    except Exception:
        pass


def reset_filters():
    st.session_state['filters'] = {}
    st.session_state['apply_clicked'] = False
    safe_rerun()


def apply_and_store(selected_countries, date_range, selected_products, search_text):
    st.session_state['filters'] = {
        'countries': selected_countries,
        'date_range': date_range,
        'products': selected_products,
        'search_text': search_text,
    }
    st.session_state['apply_clicked'] = True

with st.sidebar:
    st.markdown('<div class="filter-card"><div class="filter-title">Filters</div><div class="filter-subtitle">Refine the dataset and focus on the audience or time range you want to analyze.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">Country</div>', unsafe_allow_html=True)
    countries = sorted(df.get('Country', pd.Series(dtype=str)).dropna().unique())
    selected_countries = st.multiselect('Country', countries, default=st.session_state['filters'].get('countries', countries), key='filter_countries', label_visibility='collapsed')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">Date range</div>', unsafe_allow_html=True)
    if 'InvoiceDate' in df.columns:
        min_date = pd.to_datetime(df['InvoiceDate']).min().date()
        max_date = pd.to_datetime(df['InvoiceDate']).max().date()
        date_range = st.date_input('Invoice Date Range', value=st.session_state['filters'].get('date_range', (min_date, max_date)), key='filter_date_range', label_visibility='collapsed')
    else:
        date_range = None
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">Products</div>', unsafe_allow_html=True)
    products = df['Description'].dropna().unique().tolist() if 'Description' in df.columns else []
    selected_products = st.multiselect('Products', products[:500], default=st.session_state['filters'].get('products', None), key='filter_products', label_visibility='collapsed')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">Description search</div>', unsafe_allow_html=True)
    search_text = st.text_input('Search Description', value=st.session_state['filters'].get('search_text', ''), key='filter_search', label_visibility='collapsed')
    st.markdown('</div>', unsafe_allow_html=True)

    col_apply, col_reset = st.columns(2)
    with col_apply:
        if st.button('Apply'):
            apply_and_store(selected_countries, date_range, selected_products, search_text)
    with col_reset:
        if st.button('Reset'):
            reset_filters()

# Top action bar
action_col1, action_col2, action_col3 = st.columns([1,1,1])
if action_col1.button('Refresh Data'):
    safe_rerun()

if action_col2.button('Export Processed CSV'):
    action_col2.download_button('Download CSV', data=df.to_csv(index=False), file_name='processed_data.csv')

if action_col3.button('Show Raw Columns'):
    st.experimental_info = st.info(list(df.columns))

# apply filters to produce filtered_df using current sidebar selections
filtered_df = df.copy()
selected_countries = st.session_state.get('filter_countries', countries)
selected_products = st.session_state.get('filter_products', None)
search_text = st.session_state.get('filter_search', '')
if 'InvoiceDate' in df.columns:
    date_range = st.session_state.get('filter_date_range', (pd.to_datetime(df['InvoiceDate']).min().date(), pd.to_datetime(df['InvoiceDate']).max().date()))
else:
    date_range = None

if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
if date_range and 'InvoiceDate' in filtered_df.columns:
    start, end = date_range
    filtered_df = filtered_df[(filtered_df['InvoiceDate'] >= pd.to_datetime(start)) & (filtered_df['InvoiceDate'] <= pd.to_datetime(end))]
if selected_products and 'Description' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Description'].isin(selected_products)]
if search_text and 'Description' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Description'].str.contains(search_text, case=False, na=False)]

# Main content using tabs
tab_overview, tab_segment, tab_forecast, tab_inventory, tab_raw = st.tabs(['Overview', 'Segmentation', 'Forecast', 'Inventory', 'Raw Data'])

with tab_overview:
    st.subheader('Overview')
    total_revenue = filtered_df['TotalPrice'].sum() if 'TotalPrice' in filtered_df.columns else 0
    total_orders = filtered_df['InvoiceNo'].nunique() if 'InvoiceNo' in filtered_df.columns else len(filtered_df)
    total_customers = filtered_df['CustomerID'].nunique() if 'CustomerID' in filtered_df.columns else len(filtered_df)
    avg_order = total_revenue/total_orders if total_orders else 0

    # KPI cards with sparklines
    c1, c2, c3, c4 = st.columns([1.5,1,1,1])

    def sparkline_figure(series, color='#0066CC'):
        if PLOTLY_AVAILABLE:
            if isinstance(series.index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(series.index):
                x = series.index
            else:
                x = list(range(len(series)))
            fig = go.Figure(go.Scatter(x=x, y=series.values, mode='lines', line=dict(color=color, width=2), fill='tozeroy'))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=60)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig
        return None

    with c1:
        st.markdown('<div class="metric-card"><div class="metric-icon">💰</div><div class="metric-title">Total Revenue</div><div class="metric-value">₹ {0:,.0f}</div><div class="metric-subtitle">Revenue captured across the current filter set</div></div>'.format(total_revenue), unsafe_allow_html=True)
        if 'InvoiceDate' in filtered_df.columns and 'TotalPrice' in filtered_df.columns:
            rev_series = filtered_df.set_index('InvoiceDate').resample('D')['TotalPrice'].sum().fillna(0)
            sp = sparkline_figure(rev_series.tail(30))
            if PLOTLY_AVAILABLE and sp is not None:
                st.plotly_chart(sp, use_container_width=True)
            else:
                st.line_chart(rev_series.tail(30))
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-icon">📦</div><div class="metric-title">Total Orders</div><div class="metric-value">{0:,}</div><div class="metric-subtitle">Distinct invoices in the current view</div></div>'.format(total_orders), unsafe_allow_html=True)
        if 'InvoiceDate' in filtered_df.columns:
            orders_series = filtered_df.set_index('InvoiceDate').resample('D')['InvoiceNo'].nunique().fillna(0)
            sp = sparkline_figure(orders_series.tail(30), color='#FF9900')
            if PLOTLY_AVAILABLE and sp is not None:
                st.plotly_chart(sp, use_container_width=True)
            else:
                st.line_chart(orders_series.tail(30))
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-icon">📈</div><div class="metric-title">Avg Order Value</div><div class="metric-value">₹ {0:,.0f}</div><div class="metric-subtitle">Average spend per order</div></div>'.format(avg_order), unsafe_allow_html=True)
        if 'InvoiceDate' in filtered_df.columns and 'TotalPrice' in filtered_df.columns and 'InvoiceNo' in filtered_df.columns:
            daily = filtered_df.set_index('InvoiceDate').resample('D').agg({'TotalPrice':'sum','InvoiceNo':'nunique'})
            aov = (daily['TotalPrice'] / daily['InvoiceNo']).fillna(0)
            sp = sparkline_figure(aov.tail(30), color='#2ca02c')
            if PLOTLY_AVAILABLE and sp is not None:
                st.plotly_chart(sp, use_container_width=True)
            else:
                st.line_chart(aov.tail(30))
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-icon">👥</div><div class="metric-title">Total Customers</div><div class="metric-value">{0:,}</div><div class="metric-subtitle">Unique customers currently in scope</div></div>'.format(total_customers), unsafe_allow_html=True)
        if 'InvoiceDate' in filtered_df.columns:
            cust_series = filtered_df.set_index('InvoiceDate').resample('D')['CustomerID'].nunique().fillna(0)
            sp = sparkline_figure(cust_series.tail(30), color='#d62728')
            if PLOTLY_AVAILABLE and sp is not None:
                st.plotly_chart(sp, use_container_width=True)
            else:
                st.line_chart(cust_series.tail(30))

    st.markdown('<div class="panel-card"><div class="filter-title">Sales Trend</div></div>', unsafe_allow_html=True)
    if 'InvoiceDate' in filtered_df.columns and 'TotalPrice' in filtered_df.columns:
        sales = filtered_df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
        if PLOTLY_AVAILABLE:
            fig = px.line(sales, x='InvoiceDate', y='TotalPrice', title='Sales Over Time')
            fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#111111')
            fig.update_xaxes(showgrid=False, zeroline=False, linecolor='#e5e7eb', tickfont=dict(color='#111111'))
            fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0', zeroline=False, linecolor='#e5e7eb', tickfont=dict(color='#111111'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plt.figure(figsize=(10,4), facecolor='#ffffff')
            ax = fig.add_subplot(111, facecolor='#ffffff')
            sales.set_index('InvoiceDate')['TotalPrice'].plot(ax=ax, color='#1f77b4')
            ax.set_facecolor('#ffffff')
            ax.grid(color='#f0f0f0', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.spines['bottom'].set_color('#e5e7eb')
            ax.spines['left'].set_color('#e5e7eb')
            ax.tick_params(colors='#111111')
            st.pyplot(fig)
    else:
        st.info('Not enough data for sales trend')

with tab_segment:
    st.subheader('Customer Segmentation')
    st.markdown(
        "Use customer purchase behavior to identify high-value, loyal, and at-risk segments. "
        "This section helps managers understand where to focus retention, loyalty, and reactivation efforts."
    )
    if 'CustomerID' in filtered_df.columns:
        snapshot_date = filtered_df['InvoiceDate'].max() if 'InvoiceDate' in filtered_df.columns else pd.Timestamp.today()
        freq_series = filtered_df.groupby('CustomerID')['InvoiceNo'].nunique() if 'InvoiceNo' in filtered_df.columns else filtered_df.groupby('CustomerID').size()
        monetary_series = filtered_df.groupby('CustomerID')['TotalPrice'].sum() if 'TotalPrice' in filtered_df.columns else pd.Series(0, index=freq_series.index)
        recency_series = filtered_df.groupby('CustomerID')['InvoiceDate'].max().apply(lambda x: (snapshot_date - x).days) if 'InvoiceDate' in filtered_df.columns else pd.Series(0, index=freq_series.index)
        rfm = pd.concat([recency_series, freq_series, monetary_series], axis=1)
        rfm.columns = ['Recency','Frequency','Monetary']
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        profile = rfm.groupby('Cluster').agg({
            'Recency':'mean',
            'Frequency':'mean',
            'Monetary':'mean',
            'Cluster':'size'
        }).rename(columns={'Cluster':'Customers'}).reset_index()
        profile['Segment'] = 'Potential'
        recency_median = profile['Recency'].median()
        frequency_median = profile['Frequency'].median()
        monetary_median = profile['Monetary'].median()
        profile.loc[
            (profile['Recency'] <= recency_median) & (profile['Frequency'] >= frequency_median) & (profile['Monetary'] >= monetary_median),
            'Segment'
        ] = 'Champions'
        profile.loc[
            (profile['Recency'] > recency_median) & (profile['Monetary'] >= monetary_median),
            'Segment'
        ] = 'At Risk'
        profile.loc[
            (profile['Recency'] <= recency_median) & (profile['Monetary'] < monetary_median),
            'Segment'
        ] = 'Potential'
        profile.loc[
            (profile['Recency'] > recency_median) & (profile['Monetary'] < monetary_median),
            'Segment'
        ] = 'New / Low Value'

        segment_map = profile.set_index('Cluster')['Segment'].to_dict()
        rfm['Segment'] = rfm['Cluster'].map(segment_map)

        profile = profile[['Cluster','Segment','Customers','Recency','Frequency','Monetary']]
        profile.columns = ['Cluster','Segment','Customers','Avg Recency','Avg Frequency','Avg Spend']
        profile['Avg Recency'] = profile['Avg Recency'].round(1)
        profile['Avg Frequency'] = profile['Avg Frequency'].round(1)
        profile['Avg Spend'] = profile['Avg Spend'].round(0)

        st.markdown('**Segment summary**')
        st.dataframe(profile)

        if PLOTLY_AVAILABLE:
            fig3 = px.scatter(
                rfm.reset_index(),
                x='Recency',
                y='Monetary',
                color='Segment',
                size='Frequency',
                title='Customer segments by Recency and Total spend',
                labels={'Recency': 'Days Since Last Purchase', 'Monetary': 'Total Spend', 'Frequency': 'Number of Orders'},
                color_discrete_map={
                    'Champions': '#1f77b4',
                    'Potential': '#2ca02c',
                    'At Risk': '#ff7f0e',
                    'New / Low Value': '#d62728'
                },
            )
            fig3.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#111111', legend_title_text='Segment')
            fig3.update_xaxes(showgrid=True, gridcolor='#f0f0f0', zeroline=False, linecolor='#e5e7eb', tickfont=dict(color='#111111'))
            fig3.update_yaxes(showgrid=True, gridcolor='#f0f0f0', zeroline=False, linecolor='#e5e7eb', tickfont=dict(color='#111111'))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig3 = plt.figure(figsize=(10,5), facecolor='#ffffff')
            ax = fig3.add_subplot(111, facecolor='#ffffff')
            sns.scatterplot(
                x='Recency',
                y='Monetary',
                hue='Segment',
                size='Frequency',
                data=rfm,
                palette={
                    'Champions': '#1f77b4',
                    'Potential': '#2ca02c',
                    'At Risk': '#ff7f0e',
                    'New / Low Value': '#d62728'
                },
                sizes=(20, 200),
                ax=ax
            )
            ax.set_xlabel('Days Since Last Purchase', color='#111111')
            ax.set_ylabel('Total Spend', color='#111111')
            ax.set_title('Customer segments', color='#111111')
            ax.grid(color='#f0f0f0', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.spines['bottom'].set_color('#e5e7eb')
            ax.spines['left'].set_color('#e5e7eb')
            ax.tick_params(colors='#111111')
            legend = ax.legend(title='Segment')
            for text in legend.get_texts():
                text.set_color('#111111')
            st.pyplot(fig3)
    else:
        st.info('No CustomerID column available for segmentation')

with tab_forecast:
    st.subheader('Demand Forecast')
    st.markdown(
        'Forecast future revenue for the next 30 days using historical sales patterns. ' 
        'This helps leadership plan inventory, marketing campaigns, and cash flow.'
    )
    if 'InvoiceDate' in filtered_df.columns and 'TotalPrice' in filtered_df.columns:
        daily = filtered_df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
        daily.columns = ['ds','y']
        model = Prophet()
        model.fit(daily)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast_period = forecast.tail(30)

        current_30_days = daily.tail(30)['y'].sum() if len(daily) >= 30 else daily['y'].sum()
        future_revenue = forecast_period['yhat'].sum()
        growth_pct = ((future_revenue - current_30_days) / current_30_days * 100) if current_30_days else 0

        if PLOTLY_AVAILABLE:
            fig4 = px.line(
                forecast_period,
                x='ds',
                y='yhat',
                title='30-Day Revenue Forecast',
                labels={'ds': 'Date', 'yhat': 'Forecast Revenue'},
                color_discrete_sequence=['#1f77b4']
            )
            fig4.add_scatter(x=forecast_period['ds'], y=forecast_period['yhat_lower'], mode='lines', line=dict(width=0), name='Lower bound', fill=None)
            fig4.add_scatter(x=forecast_period['ds'], y=forecast_period['yhat_upper'], mode='lines', line=dict(width=0), name='Upper bound', fill='tonexty', fillcolor='rgba(31,119,180,0.2)')
            fig4.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#111111')
            fig4.update_xaxes(showgrid=True, gridcolor='#f0f0f0', zeroline=False, linecolor='#e5e7eb', tickfont=dict(color='#111111'))
            fig4.update_yaxes(showgrid=True, gridcolor='#f0f0f0', zeroline=False, linecolor='#e5e7eb', tickfont=dict(color='#111111'))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            fig4 = plt.figure(figsize=(10,5), facecolor='#ffffff')
            ax = fig4.add_subplot(111, facecolor='#ffffff')
            ax.plot(forecast_period['ds'], forecast_period['yhat'], color='#1f77b4', label='Forecast')
            ax.fill_between(forecast_period['ds'], forecast_period['yhat_lower'], forecast_period['yhat_upper'], color='#1f77b4', alpha=0.2)
            ax.set_title('30-Day Revenue Forecast', color='#111111')
            ax.set_xlabel('Date', color='#111111')
            ax.set_ylabel('Forecast Revenue', color='#111111')
            ax.grid(color='#f0f0f0', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.spines['bottom'].set_color('#e5e7eb')
            ax.spines['left'].set_color('#e5e7eb')
            ax.tick_params(colors='#111111')
            ax.legend()
            st.pyplot(fig4)

        colf1, colf2 = st.columns(2)
        colf1.metric('Forecasted 30-day revenue', f'₹ {future_revenue:,.0f}')
        colf2.metric('30-day trend vs recent period', f'{growth_pct:,.1f}%')
    else:
        st.info('Not enough data to run forecasting')

with tab_inventory:
    st.subheader('Inventory Recommendation')
    if 'forecast' in locals():
        st.metric('Recommended Stock (Next 30 Days)', f"{forecast['yhat'].tail(30).sum():,.0f}")
    else:
        st.info('Run Forecast to see inventory recommendations')

with tab_raw:
    st.subheader('Raw Data')
    # global search
    if 'raw_search' not in st.session_state:
        st.session_state['raw_search'] = ''
    raw_search = st.text_input('Search table (global)', value=st.session_state['raw_search'])
    st.session_state['raw_search'] = raw_search

    df_display = filtered_df.copy()
    if raw_search:
        str_cols = df_display.select_dtypes(include=['object', 'string']).columns
        if len(str_cols):
            combined = df_display[str_cols].fillna('').agg(' '.join, axis=1)
            df_display = df_display[combined.str.contains(raw_search, case=False, na=False)]

    # pagination
    page_size = 50
    total_rows = len(df_display)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    if 'raw_page' not in st.session_state:
        st.session_state['raw_page'] = 1
    colp1, colp2, colp3 = st.columns([1,1,6])
    if colp1.button('Previous') and st.session_state['raw_page']>1:
        st.session_state['raw_page'] -= 1
    if colp2.button('Next') and st.session_state['raw_page']<total_pages:
        st.session_state['raw_page'] += 1
    start = (st.session_state['raw_page']-1)*page_size
    end = start + page_size
    st.markdown(f'Page {st.session_state["raw_page"]} / {total_pages} — Showing rows {start+1} to {min(end,total_rows)} of {total_rows}')
    st.dataframe(df_display.iloc[start:end])

    if st.button('Save UI Preferences'):
        set_pref('raw_page', st.session_state['raw_page'])
        set_pref('raw_page_size', page_size)
        st.success('Preferences saved')

# download processed filtered dataset
st.download_button('Download Filtered Data', data=filtered_df.to_csv(index=False), file_name='filtered_data.csv')
