import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="RetailPulse - AI Retail Analytics", layout="wide")

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("📊 RetailPulse - AI Powered Retail Analytics Dashboard")
st.markdown("An end-to-end Data Science and Analytics platform for demand forecasting, customer segmentation, churn analysis, and inventory optimization.")


def preprocess_retail_data(df):
    """Normalize common retail column names and create required analytics fields."""
    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    df = df.copy()
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    customer_candidates = ["customerid", "customer id", "customer_id", "customer", "custid", "cust_id"]
    customer_col = next((cols_lower[c] for c in customer_candidates if c in cols_lower), None)
    if customer_col is None:
        df["CustomerID"] = [f"CUST_{i}" for i in range(len(df))]
    else:
        df = df.rename(columns={customer_col: "CustomerID"})

    invoice_candidates = ["invoiceno", "invoice no", "invoice_number", "invoice", "orderid", "order id", "order_no", "order"]
    invoice_col = next((cols_lower[c] for c in invoice_candidates if c in cols_lower), None)
    if invoice_col is not None:
        df = df.rename(columns={invoice_col: "InvoiceNo"})
    elif "InvoiceNo" not in df.columns:
        df["InvoiceNo"] = [f"INV_{i}" for i in range(len(df))]

    date_candidates = ["invoicedate", "invoice date", "date", "transactiondate"]
    date_col = next((cols_lower[c] for c in date_candidates if c in cols_lower), None)
    if date_col is not None:
        df["InvoiceDate"] = pd.to_datetime(df[date_col], errors="coerce")
    elif "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    else:
        df["InvoiceDate"] = pd.NaT

    qty_candidates = ["quantity", "qty"]
    price_candidates = ["unitprice", "unit price", "price", "unit_price"]
    desc_candidates = ["description", "product", "item"]

    qty_col = next((cols_lower[c] for c in qty_candidates if c in cols_lower), None)
    price_col = next((cols_lower[c] for c in price_candidates if c in cols_lower), None)
    desc_col = next((cols_lower[c] for c in desc_candidates if c in cols_lower), None)

    if qty_col is not None:
        df = df.rename(columns={qty_col: "Quantity"})
    elif "Quantity" not in df.columns:
        df["Quantity"] = np.nan

    if price_col is not None:
        df = df.rename(columns={price_col: "UnitPrice"})
    elif "UnitPrice" not in df.columns:
        if "Price" in df.columns:
            df = df.rename(columns={"Price": "UnitPrice"})
        else:
            df["UnitPrice"] = np.nan

    if desc_col is not None:
        df = df.rename(columns={desc_col: "Description"})
    elif "Description" not in df.columns and "Product" in df.columns:
        df = df.rename(columns={"Product": "Description"})
    elif "Description" not in df.columns:
        df["Description"] = "Unknown Product"

    if "TotalPrice" not in df.columns:
        if "Quantity" in df.columns and "UnitPrice" in df.columns:
            df["TotalPrice"] = pd.to_numeric(df["Quantity"], errors="coerce") * pd.to_numeric(df["UnitPrice"], errors="coerce")
        elif "Sales" in df.columns:
            df["TotalPrice"] = pd.to_numeric(df["Sales"], errors="coerce")
        else:
            df["TotalPrice"] = 0.0
    else:
        df["TotalPrice"] = pd.to_numeric(df["TotalPrice"], errors="coerce")

    df["Quantity"] = pd.to_numeric(df.get("Quantity", np.nan), errors="coerce").fillna(0)
    df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice", np.nan), errors="coerce").fillna(0)
    df["TotalPrice"] = pd.to_numeric(df.get("TotalPrice", 0), errors="coerce").fillna(0)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["InvoiceDate"] = df["InvoiceDate"].fillna(pd.Timestamp.today())

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].fillna("Unknown Customer").astype(str)

    return df


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    [
        "📁 Upload Dataset",
        "📊 Sales Analytics",
        "👥 Customer Segmentation",
        "📈 Demand Forecasting",
        "⚠️ Churn Prediction",
        "📦 Inventory Optimization",
        "📑 Project Summary"
    ]
)

# -------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------
if page == "📁 Upload Dataset":
    st.header("Upload Retail Dataset")

    uploaded_file = st.file_uploader("Upload your retail dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = preprocess_retail_data(df)
        st.session_state["data"] = df
        st.success("Dataset uploaded successfully and standardized for analysis!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")
        st.write(df.describe(include="all"))

# -------------------------------------------------
# LOAD DATA FROM SESSION
# -------------------------------------------------
if "data" in st.session_state:
    df = preprocess_retail_data(st.session_state["data"])
    st.session_state["data"] = df

# -------------------------------------------------
# SALES ANALYTICS PAGE
# -------------------------------------------------
if page == "📊 Sales Analytics" and "data" in st.session_state:
    st.header("Sales Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    total_revenue = df["TotalPrice"].sum() if "TotalPrice" in df.columns else 0
    total_orders = df["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else len(df)
    total_customers = df["CustomerID"].nunique() if "CustomerID" in df.columns else len(df)

    col1.metric("Total Revenue", f"₹ {total_revenue:,.0f}")
    col2.metric("Total Orders", total_orders)
    col3.metric("Total Customers", total_customers)

    st.subheader("Daily Sales Trend")

    if "InvoiceDate" in df.columns and "TotalPrice" in df.columns:
        daily_sales = df.groupby(df["InvoiceDate"].dt.normalize())["TotalPrice"].sum()
        fig = plt.figure()
        daily_sales.plot()
        plt.title("Daily Sales Trend")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        st.pyplot(fig)
    else:
        st.info("Invoice date and revenue columns are required for the trend chart.")

    st.subheader("Top 10 Products")

    if "Description" in df.columns and "Quantity" in df.columns:
        top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        fig2 = plt.figure()
        top_products.plot(kind="bar")
        plt.title("Top 10 Selling Products")
        st.pyplot(fig2)
    else:
        st.info("Product descriptions are missing from the uploaded data.")

# -------------------------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------------------------
if page == "👥 Customer Segmentation" and "data" in st.session_state:
    st.header("Customer Segmentation using RFM + KMeans")

    if "CustomerID" in df.columns and "InvoiceDate" in df.columns:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "count",
            "TotalPrice": "sum"
        })

        rfm.columns = ["Recency", "Frequency", "Monetary"]
        rfm = rfm.fillna(0)

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

        st.subheader("Customer Segmentation Table")
        st.dataframe(rfm.head())

        st.subheader("Cluster Visualization")

        fig3 = plt.figure()
        sns.scatterplot(x="Recency", y="Monetary", hue="Cluster", data=rfm)
        st.pyplot(fig3)
    else:
        st.info("Customer segmentation requires CustomerID and InvoiceDate columns.")

# -------------------------------------------------
# DEMAND FORECASTING
# -------------------------------------------------
if page == "📈 Demand Forecasting" and "data" in st.session_state:
    st.header("Demand Forecasting (30 Days)")

    if "InvoiceDate" in df.columns and "TotalPrice" in df.columns:
        daily_sales = df.groupby(df["InvoiceDate"].dt.normalize())["TotalPrice"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]
        daily_sales = daily_sales.sort_values("ds")

        try:
            model = Prophet()
            model.fit(daily_sales)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            st.subheader("Forecast Table")
            st.dataframe(forecast[["ds", "yhat"]].tail(30))

            st.subheader("Forecast Plot")
            fig4 = model.plot(forecast)
            st.pyplot(fig4)
        except Exception as exc:
            st.error(f"Forecasting could not be completed: {exc}")
    else:
        st.info("Forecasting requires InvoiceDate and TotalPrice columns.")

# -------------------------------------------------
# CHURN PREDICTION
# -------------------------------------------------
if page == "⚠️ Churn Prediction" and "data" in st.session_state:
    st.header("Customer Churn Prediction")

    if "CustomerID" in df.columns and "InvoiceDate" in df.columns:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
        churn = (snapshot_date - last_purchase).dt.days > 90

        churn_df = churn.reset_index()
        churn_df.columns = ["CustomerID", "Churn"]
        churn_count = churn_df["Churn"].value_counts()

        fig5 = plt.figure()
        churn_count.plot(kind="bar")
        plt.title("Churn vs Active Customers")
        st.pyplot(fig5)

        st.subheader("Churn Table")
        st.dataframe(churn_df.head())
    else:
        st.info("Churn analysis requires CustomerID and InvoiceDate columns.")

# -------------------------------------------------
# INVENTORY OPTIMIZATION
# -------------------------------------------------
if page == "📦 Inventory Optimization" and "data" in st.session_state:
    st.header("Inventory Optimization Recommendation")

    if "InvoiceDate" in df.columns and "TotalPrice" in df.columns:
        daily_sales = df.groupby(df["InvoiceDate"].dt.normalize())["TotalPrice"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]
        daily_sales = daily_sales.sort_values("ds")

        try:
            model = Prophet()
            model.fit(daily_sales)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            recommended_stock = forecast["yhat"].tail(30).sum()
            st.metric("Recommended Stock for Next 30 Days", f"₹ {recommended_stock:,.0f}")
        except Exception as exc:
            st.error(f"Inventory recommendation could not be completed: {exc}")
    else:
        st.info("Inventory optimization requires InvoiceDate and TotalPrice columns.")

# -------------------------------------------------
# PROJECT SUMMARY
# -------------------------------------------------
if page == "📑 Project Summary":
    st.header("Project Overview")

    st.markdown("""
    ### This dashboard includes:

    ✔ Sales analytics
    ✔ Customer segmentation (RFM + KMeans)
    ✔ Demand forecasting (Prophet Model)
    ✔ Customer churn detection
    ✔ Inventory optimization recommendations

    ### Technologies Used:

    - Python
    - Pandas & NumPy
    - Scikit-learn
    - Prophet
    - Streamlit
    - Data Visualization (Matplotlib, Seaborn)
    """)

    st.success("This is a complete end-to-end data science project ready for portfolio submission.")
