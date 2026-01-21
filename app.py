import streamlit as st
import pandas as pd
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load and Prepare Data


# --- Page Config ---
st.set_page_config(layout="wide", page_title="King County Housing Dashboard")

@st.cache_data
def load_and_model_data():
    df = pd.read_csv("kc_house_data.csv")
    
    # K-Means Modeling
    features = ['price', 'sqft_living', 'grade', 'sqft_living15', 'bathrooms', 'view', 'lat']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster_raw'] = kmeans.fit_predict(X_scaled)
    
    # Sort for Intuitive Order
    cluster_order = df.groupby('cluster_raw')['price'].mean().sort_values().index
    mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_order)}
    df['cluster'] = df['cluster_raw'].map(mapping)
    
    cluster_names = {0: "Budget", 1: "Affordable", 2: "Mid-Range", 3: "Upscale", 4: "Top tier"}
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    return df

df = load_and_model_data()

# --- Dashboard Header ---
st.title("🏡 King County Real Estate Insights")
st.markdown("### Interactive Market Segmentation Dashboard")
st.info("💡 **Brushing & Linking:** Click and drag on the Map or Scatter Plot to filter the bottom charts.")

# --- Selection Tool (The "Link") ---
brush = alt.selection_interval(resolve='global')

# Color Palette (Okabe-Ito)
color_scale = alt.Scale(
    domain=['Budget', 'Affordable', 'Mid-Range', 'Upscale', 'Top tier'],
    range=['#009E73', '#56B4E9', '#F0E442', '#0072B2', '#D55E00']
)

# --- Charts ---

# 1. Geographic Map
map_chart = alt.Chart(df.sample(2500)).mark_circle(size=15).encode(
    x=alt.X('long:Q', scale=alt.Scale(domain=[df['long'].min(), df['long'].max()])),
    y=alt.Y('lat:Q', scale=alt.Scale(domain=[df['lat'].min(), df['lat'].max()])),
    color=alt.condition(brush, 'cluster_name:N', alt.value('lightgray'), scale=color_scale, title="Market Tier"),
    tooltip=['price', 'sqft_living', 'grade']
).properties(height=350, title="Geographic Distribution (Select Area)").add_selection(brush)

# 2. Price vs Size Scatter
scatter_chart = alt.Chart(df.sample(2500)).mark_circle(size=30).encode(
    x=alt.X('sqft_living:Q', title="Living Space (Sqft)"),
    y=alt.Y('price:Q', title="Price ($)"),
    color=alt.condition(brush, 'cluster_name:N', alt.value('lightgray'), scale=color_scale),
    tooltip=['price', 'sqft_living', 'grade']
).properties(height=350, title="Price vs. Size").add_selection(brush)

# 3. Bar Chart (Filtered by Brush)
bar_chart = alt.Chart(df.sample(2500)).mark_bar().encode(
    x=alt.X('grade:O', title="Construction Grade"),
    y=alt.Y('count():Q', title="Houses"),
    color=alt.Color('cluster_name:N', scale=color_scale)
).transform_filter(brush).properties(height=250, title="Inventory by Grade")

# 4. Histogram (Filtered by Brush)
hist_chart = alt.Chart(df.sample(2500)).mark_bar().encode(
    x=alt.X('yr_built:Q', bin=alt.Bin(maxbins=30), title="Year Built"),
    y=alt.Y('count():Q'),
    color=alt.Color('cluster_name:N', scale=color_scale)
).transform_filter(brush).properties(height=250, title="Construction Trends")

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.altair_chart(map_chart, use_container_width=True)
    st.altair_chart(bar_chart, use_container_width=True)

with col2:
    st.altair_chart(scatter_chart, use_container_width=True)
    st.altair_chart(hist_chart, use_container_width=True)