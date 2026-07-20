# NeuralRetail Dashboard

This is a production-ready Streamlit retail analytics dashboard for executive decision-making.

## What it does
- Loads retail sales data from `data/raw/online_retail.xlsx` or the first CSV/XLSX found in `data/raw/`
- Standardizes key retail columns and computes `TotalPrice`
- Provides filters for Country, Date Range, Product, and full-text product search
- Displays manager-focused KPIs and interactive visualizations
- Segments customers into Champions, Potential, At Risk, and New/Low Value
- Forecasts revenue for the next 30 days with confidence intervals
- Provides inventory recommendations and raw data pagination

## Run locally
```bash
cd c:\Users\palak\intern_course
c:\Users\palak\intern_course\.venv\Scripts\python.exe -m streamlit run NeuralRetail_app.py
```

## Requirements
Install dependencies:
```bash
c:\Users\palak\intern_course\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Production readiness notes
- Uses a local `.streamlit_prefs.json` file for simple UI persistence
- Uses caching for data loading via `@st.cache_data`
- Fits Prophet for fast 30-day forecasting
- Includes filter-driven KPIs, segmentation and forecast summary cards
- Deployment options:
  - Streamlit Cloud
  - Azure App Service / AWS Elastic Beanstalk targeting Python 3.13
  - Dockerize with the workspace virtualenv and `requirements.txt` for repeated deployments
