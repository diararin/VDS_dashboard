import streamlit as st
import pandas as pd
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Config for Single-Screen View ---
st.set_page_config(layout="wide", page_title="King County Housing Dashboard")

@st.cache_data
def get_processed_data():
    df = pd.read_csv('kc_house_data.csv')
    
    # Feature Selection & Scaling for K-Means
    features = ['price', 'sqft_living', 'grade', 'sqft_living15', 'bathrooms', 'view', 'lat']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modeling & Intuitive Sorting
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster_raw'] = kmeans.fit_predict(X_scaled)
    order = df.groupby('cluster_raw')['price'].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    df['tier'] = df['cluster_raw'].map(mapping).map({
        0: "Budget", 1: "Affordable", 2: "Mid-Range", 3: "Upscale", 4: "Top Tier"
    })
    return df

df = get_processed_data()
# Sample for smooth browser performance (Brushing & Linking is faster with sampling)
source = df.sample(2500, random_state=42)

# --- Visual Encoding & Selection ---
# Okabe-Ito Palette (5 Colors, Colorblind-Safe)
okabe_ito = ['#009E73', '#56B4E9', '#F0E442', '#0072B2', '#D55E00']
color_scale = alt.Scale(
    domain=["Budget", "Affordable", "Mid-Range", "Upscale", "Top Tier"],
    range=okabe_ito
)

# Define the Global Brush for linking
brush = alt.selection_interval(empty='all')

# --- Row 1: The Interactive Controllers ---
map_chart = alt.Chart(source).mark_circle(size=15).encode(
    x=alt.X('long:Q', scale=alt.Scale(domain=[df.long.min(), df.long.max()])),
    y=alt.Y('lat:Q', scale=alt.Scale(domain=[df.lat.min(), df.lat.max()])),
    color=alt.condition(brush, 'tier:N', alt.value('lightgray'), scale=color_scale, title="Market Tier"),
    tooltip=['price', 'sqft_living', 'tier']
).properties(width=380, height=280, title="1. Map (Select Area to Filter)").add_selection(brush)

scatter = alt.Chart(source).mark_circle(size=30).encode(
    x=alt.X('sqft_living:Q', title="Living Space (sqft)"),
    y=alt.Y('price:Q', title="Price ($)"),
    color=alt.condition(brush, 'tier:N', alt.value('lightgray'), scale=color_scale),
    tooltip=['price', 'grade']
).properties(width=380, height=280, title="2. Price vs. Size (Select to Filter)").add_selection(brush)

# --- Row 2: Quality & Age Distribution ---
bars = alt.Chart(source).mark_bar(opacity=0.8).encode(
    x=alt.X('grade:O', title="Construction Grade"),
    y=alt.Y('count():Q', title="Houses"),
    color=alt.Color('tier:N', scale=color_scale)
).transform_filter(brush).properties(width=380, height=200, title="3. Quality Grades in Selection")

hist = alt.Chart(source).mark_bar(opacity=0.8).encode(
    x=alt.X('yr_built:Q', bin=alt.Bin(maxbins=25), title="Year Built"),
    y=alt.Y('count():Q'),
    color=alt.Color('tier:N', scale=color_scale)
).transform_filter(brush).properties(width=380, height=200, title="4. Build Year Trends")

# --- Row 3: New Utility Distributions ---
bed_chart = alt.Chart(source).mark_bar(opacity=0.8).encode(
    x=alt.X('bedrooms:O', title="Bedrooms"),
    y=alt.Y('count():Q', title="Count"),
    color=alt.Color('tier:N', scale=color_scale)
).transform_filter(brush).properties(width=380, height=200, title="5. Bedrooms Distribution")

bath_chart = alt.Chart(source).mark_bar(opacity=0.8).encode(
    x=alt.X('bathrooms:O', title="Bathrooms"),
    y=alt.Y('mean(price):Q', title="Avg Price ($)"),
    color=alt.Color('tier:N', scale=color_scale)
).transform_filter(brush).properties(width=380, height=200, title="6. Avg Price by Bathroom Count")

# --- Combine into Dashboard Object ---
st.title("🏡 King County Real Estate Dashboard (6-View)")
st.info("💡 **Interactive Insight:** Use your mouse to draw a box on the Map or Scatter Plot. All 6 charts are linked.")

# Altair horizontal and vertical concatenation
dashboard = (map_chart | scatter) & (bars | hist) & (bed_chart | bath_chart)

st.altair_chart(dashboard, use_container_width=True)
