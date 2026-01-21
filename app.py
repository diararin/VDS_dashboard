import streamlit as st
import pandas as pd
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

@st.cache_data
def get_data():
    df = pd.read_csv('kc_house_data.csv')
    # Modeling
    features = ['price', 'sqft_living', 'grade', 'sqft_living15', 'bathrooms', 'view', 'lat']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    # Sort clusters by price
    order = df.groupby('cluster_id')['price'].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    df['tier'] = df['cluster_id'].map(mapping).map({
        0: "Budget", 1: "Affordable", 2: "Mid-Range", 3: "Upscale", 4: "Top Tier"
    })
    return df

df = get_data()

# IMPORTANT: Sample the data once so all charts share the exact same points
source = df.sample(2000, random_state=42)

st.title("King County Real Estate Dashboard")
st.write("Click and drag on the Map or Scatter Plot to filter the charts below.")

# 1. Define the Selection (The Link)
# empty='all' ensures charts aren't blank at the start
brush = alt.selection_interval(empty='all')

# 2. Define Color Scale
color_scale = alt.Scale(
    domain=["Budget", "Affordable", "Mid-Range", "Upscale", "Top Tier"],
    range=['#009E73', '#56B4E9', '#F0E442', '#0072B2', '#D55E00']
)

# 3. Create Charts
# Top Left: Map
map_chart = alt.Chart(source).mark_circle(size=15).encode(
    x=alt.X('long:Q', scale=alt.Scale(domain=[df.long.min(), df.long.max()])),
    y=alt.Y('lat:Q', scale=alt.Scale(domain=[df.lat.min(), df.lat.max()])),
    color=alt.condition(brush, 'tier:N', alt.value('lightgray'), scale=color_scale),
    tooltip=['price', 'sqft_living']
).properties(width=400, height=300, title="Map (Select Area)").add_selection(brush)

# Top Right: Scatter
scatter = alt.Chart(source).mark_circle(size=30).encode(
    x=alt.X('sqft_living:Q', title="Size (Sqft)"),
    y=alt.Y('price:Q', title="Price ($)"),
    color=alt.condition(brush, 'tier:N', alt.value('lightgray'), scale=color_scale),
    tooltip=['price', 'grade']
).properties(width=400, height=300, title="Price vs. Size").add_selection(brush)

# Bottom Left: Grades (Filtered)
bars = alt.Chart(source).mark_bar().encode(
    x=alt.X('grade:O', title="Grade"),
    y=alt.Y('count():Q'),
    color=alt.Color('tier:N', scale=color_scale)
).transform_filter(brush).properties(width=400, height=200, title="Quality Grades in Selection")

# Bottom Right: Year Built (Filtered)
hist = alt.Chart(source).mark_bar().encode(
    x=alt.X('yr_built:Q', bin=alt.Bin(maxbins=30), title="Year Built"),
    y=alt.Y('count():Q'),
    color=alt.Color('tier:N', scale=color_scale)
).transform_filter(brush).properties(width=400, height=200, title="Construction Year in Selection")

# 4. Combine into a single dashboard layout
# This ensures brushing/linking works across all views
dashboard = (map_chart | scatter) & (bars | hist)

# Display in Streamlit
st.altair_chart(dashboard, use_container_width=True)
