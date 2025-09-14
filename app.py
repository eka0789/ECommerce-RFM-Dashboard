import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Custom CSS for modern look with background gradient and styling
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: #ffffff;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    /* Header and text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .stMarkdown p {
        color: #e0e0e0;
    }
    /* Button and selectbox styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox>div {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    /* Dataframe styling */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    /* Metric cards */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="E-Commerce RFM Dashboard", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("🚀 E-Commerce RFM Analytics Dashboard")
st.markdown("""
Welcome to a cutting-edge, interactive dashboard for e-commerce insights!  
Sourced from [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data), this RFM (Recency, Frequency, Monetary) analysis segments customers to drive strategic decisions.  
Dive in, explore, and visualize—crafted to impress and inspire!
""")

# Cache data loading
@st.cache_data
def load_data():
    try:
        file_id = '1XYXRu-MR2b_r5cF5_mwe_qXOiQA6HFNX'
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Error fetching file: Status {response.status_code}. Ensure file is public!")
            return pd.DataFrame()
        if '<html' in response.text.lower() or 'drive.google.com' in response.text.lower():
            st.error("File returned HTML, not CSV. Set file to 'Anyone with the link' in Google Drive.")
            return pd.DataFrame()
        df = pd.read_csv(StringIO(response.text), encoding='latin1', encoding_errors='ignore', low_memory=False)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'].notna())]
        st.success(f"Data loaded! Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Check Google Drive sharing settings.")
        return pd.DataFrame()

# Load dataset
df = load_data()
if df.empty:
    st.stop()

# Enhanced Sidebar
st.sidebar.header("⚙️ Filters & Options")
country_filter = st.sidebar.selectbox("Select Country 🌍", options=["All"] + sorted(list(df['Country'].unique())))
min_date = df['InvoiceDate'].min()
max_date = df['InvoiceDate'].max()
date_range = st.sidebar.date_input("Select Date Range 📅", [min_date, max_date], min_value=min_date, max_value=max_date)
num_clusters = st.sidebar.slider("Number of Clusters 🔢", min_value=4, max_value=6, value=4, step=1)  # Min 4 to match labels

# Filter data
filtered_df = df.copy()
if country_filter != "All":
    filtered_df = filtered_df[filtered_df['Country'] == country_filter]
if date_range and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['InvoiceDate'] >= start_date) & (filtered_df['InvoiceDate'] <= end_date)]

# Key Metrics
st.header("📊 At-a-Glance Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Revenue 💰", f"£{filtered_df['TotalPrice'].sum():,.2f}")
with col2:
    st.metric("Total Customers 👥", f"{filtered_df['CustomerID'].nunique()}")
with col3:
    st.metric("Total Invoices 📑", f"{filtered_df['InvoiceNo'].nunique()}")
with col4:
    st.metric("Avg. Order Value 🛒", f"£{filtered_df['TotalPrice'].sum() / filtered_df['InvoiceNo'].nunique():,.2f}")

# Dataset Exploration
with st.expander("🔍 Explore Raw Data", expanded=False):
    st.write("Random sample of 10 records:")
    st.dataframe(filtered_df.sample(10, random_state=123).style.background_gradient(cmap='viridis'))

# RFM Analysis
st.header("📈 RFM Analysis & Segmentation")
today = dt.date(2012, 1, 1)
rfm = filtered_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (today - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
rfm = rfm[rfm['Monetary'] > 0]

if not rfm.empty:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Elbow Method
    with st.expander("Optimal Cluster Analysis 📉", expanded=False):
        sse = []
        max_k = min(6, len(rfm) - 1)
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(rfm_scaled)
            sse.append(kmeans.inertia_)
        fig_elbow, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), sse, marker='o', color='cyan', linestyle='--')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('SSE')
        ax.set_title('Elbow Method')
        ax.grid(True)
        ax.set_facecolor((1, 1, 1, 0.1))  # Fixed RGBA tuple
        fig_elbow.patch.set_facecolor((1, 1, 1, 0.1))  # Fixed background
        st.pyplot(fig_elbow)

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Labeling - Fixed to use specific labels for 4 clusters, extend if more
    avg_rfm = rfm.groupby('Cluster').mean(numeric_only=True).reset_index()
    avg_rfm['Score'] = -avg_rfm['Recency'] + avg_rfm['Frequency'] + (avg_rfm['Monetary'] / 1000)
    avg_rfm = avg_rfm.sort_values('Score', ascending=False)
    labels = ['Champions 🏆', 'Loyal Customers ❤️', 'At Risk ⚠️', 'New Customers 🌱']
    if num_clusters > 4:
        labels += [f'Additional Segment {i+1} 🔍' for i in range(num_clusters - 4)]
    label_map = {avg_rfm.iloc[i]['Cluster']: labels[i] for i in range(num_clusters)}
    rfm['Segment'] = rfm['Cluster'].map(label_map)
    avg_rfm['Segment'] = avg_rfm['Cluster'].map(label_map)

    # Enhanced Visuals with more variety
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RFM Metrics per Segment 📊")
        avg_rfm_melt = pd.melt(avg_rfm, id_vars=['Cluster', 'Segment'], value_vars=['Recency', 'Frequency', 'Monetary'])
        fig_rfm = px.bar(avg_rfm_melt, x='Segment', y='value', color='variable', barmode='group',
                         title='Average RFM Values',
                         color_discrete_map={'Recency': '#FF6347', 'Frequency': '#32CD32', 'Monetary': '#1E90FF'})
        fig_rfm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                              legend_title_text='Metric', hovermode='x unified')
        fig_rfm.update_traces(marker_line_width=1.5, opacity=0.85)
        st.plotly_chart(fig_rfm, use_container_width=True)

    with col2:
        st.subheader("Customer Distribution 🧑‍🤝‍🧑")
        customer_dist = rfm['Segment'].value_counts().reset_index()
        customer_dist.columns = ['Segment', 'Count']
        fig_dist = px.pie(customer_dist, values='Count', names='Segment', title='Customer Segments',
                          color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)  # Donut chart for modern look
        fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        fig_dist.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value')
        st.plotly_chart(fig_dist, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Revenue by Segment 💸")
        revenue_contrib = rfm.groupby('Segment')['Monetary'].sum().reset_index()
        fig_revenue = px.bar(revenue_contrib, x='Segment', y='Monetary', title='Revenue Contribution',
                             color='Segment', color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_revenue.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                  showlegend=False)
        fig_revenue.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.85)
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col4:
        st.subheader("RFM Scatter Plot 🔍")
        fig_scatter = px.scatter(rfm.reset_index(), x='Recency', y='Frequency', size='Monetary', color='Segment',
                                 title='RFM Cluster Distribution', hover_name='CustomerID',
                                 color_discrete_sequence=px.colors.qualitative.Set1)
        fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_scatter, use_container_width=True)

# Revenue Trends
st.header("📅 Revenue Trends")
filtered_df_with_dt = filtered_df.copy()
filtered_df_with_dt['InvoiceDate_dt'] = pd.to_datetime(filtered_df['InvoiceDate'])
filtered_df_with_dt['Month'] = filtered_df_with_dt['InvoiceDate_dt'].dt.to_period('M').astype(str)
monthly_revenue = filtered_df_with_dt.groupby('Month')['TotalPrice'].sum().reset_index()
fig_monthly = px.line(monthly_revenue, x='Month', y='TotalPrice', title='Monthly Revenue',
                      line_shape='spline', markers=True, color_discrete_sequence=['#FFD700'])
fig_monthly.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
fig_monthly.update_traces(line_width=3, hovertemplate='Month: %{x}<br>Revenue: £%{y:,.2f}')
st.plotly_chart(fig_monthly, use_container_width=True)

# Additional Visualizations: Top Products
st.header("🛍️ Product Insights")
top_products = filtered_df.groupby('Description')['TotalPrice'].sum().reset_index().sort_values('TotalPrice', ascending=False).head(10)
fig_top_products = px.bar(top_products, x='TotalPrice', y='Description', orientation='h',
                          title='Top 10 Products by Revenue',
                          color='TotalPrice', color_continuous_scale=px.colors.sequential.Viridis)
fig_top_products.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                               yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_top_products, use_container_width=True)

# Additional: Quantity Sold by Product
top_quantity = filtered_df.groupby('Description')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False).head(10)
fig_top_quantity = px.bar(top_quantity, x='Quantity', y='Description', orientation='h',
                          title='Top 10 Products by Quantity Sold',
                          color='Quantity', color_continuous_scale=px.colors.sequential.Inferno)
fig_top_quantity.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                               yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_top_quantity, use_container_width=True)

# Geographic Insights
st.header("🌍 Global Sales Insights")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Revenue by Country (Map) 🗺️")
    df_country = filtered_df.groupby('Country')['TotalPrice'].sum().reset_index()
    fig_geo = px.choropleth(df_country, locations='Country', locationmode='country names', color='TotalPrice',
                            hover_name='Country', title='Global Revenue',
                            color_continuous_scale=px.colors.sequential.Plasma)
    fig_geo.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_geo, use_container_width=True)

with col2:
    st.subheader("Revenue by Country (Bar) 📊")
    fig_country_bar = px.bar(df_country.sort_values('TotalPrice', ascending=False),
                             x='Country', y='TotalPrice', title='Top Countries',
                             color='TotalPrice', color_continuous_scale=px.colors.sequential.Magma)
    fig_country_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_country_bar, use_container_width=True)

# Additional: Treemap for Country Revenue
fig_treemap = px.treemap(df_country, path=['Country'], values='TotalPrice', title='Revenue Treemap by Country',
                         color='TotalPrice', color_continuous_scale=px.colors.sequential.Blues)
fig_treemap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
st.plotly_chart(fig_treemap, use_container_width=True)

# Conclusions
st.header("📝 Insights & Recommendations")
st.markdown("""
**Key Insights:**  
- **Champions Rule:** Top customers drive revenue—keep them happy!  
- **Retention Gap:** New customers abound, but repeat purchases lag—boost loyalty.  
- **Product Patterns:** Bundle collectible items for upsells.  
- **Global Potential:** UK leads, but Germany and Netherlands show promise.

**Action Plan:**  
1. Offer exclusive perks to Champions.  
2. Re-engage At Risk with targeted campaigns.  
3. Bundle related products for collectors.  
4. Expand internationally with promotions like free shipping.

Ready to transform your business with data-driven insights? Let's collaborate!
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data: Kaggle | [Hire Me for Custom Dashboards!](https://myportfoliofsdev.vercel.app/)")