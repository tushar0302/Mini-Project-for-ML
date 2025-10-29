# Devpunya Customer Segmentation Dashboard
# Developed by: Tushar Joshi


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#  PAGE CONFIGURATION
st.set_page_config(page_title="Devpunya Customer Segmentation", layout="centered")

st.title("🔮 Devpunya Customer Segmentation Dashboard")
st.markdown("""
Welcome to the **Devpunya Customer Segmentation System**  
This dashboard uses **Machine Learning (K-Means Clustering)** to analyze customer behavior
and identify segments based on their service usage patterns.
""")

#  LOAD DATASET 
try:
    df = pd.read_csv("customer_segments.csv")
    st.success(" Using local dataset: customer_segments.csv")
except FileNotFoundError:
    st.error(" Dataset not found! Please ensure 'customer_segments.csv' is in the same folder.")
    st.stop()

# SHOW DATA PREVIEW 
st.subheader(" Customer Data Preview")
st.dataframe(df.head())

st.info(f"Total Records: {df.shape[0]} | Total Columns: {df.shape[1]}")

# DATA CLEANING 
df = df.dropna()
st.write(" Missing values removed (if any).")

#  SELECT FEATURES 
st.subheader(" Feature Selection for Clustering")
required_columns = ['Bookings_per_Year', 'Average_Spending', 'Satisfaction_Rating']

# Check if required columns exist
for col in required_columns:
    if col not in df.columns:
        st.error(f" Column '{col}' not found in dataset!")
        st.stop()

features = df[required_columns]

# DATA SCALING
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#  MODEL TRAINING 
st.subheader(" Applying K-Means Clustering")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

st.success(" K-Means clustering applied successfully!")

# CLUSTER SUMMARY 
st.subheader(" Cluster Summary")
summary = df.groupby('Cluster')[required_columns].mean()
st.dataframe(summary.style.highlight_max(axis=0, color='lightgreen'))

# VISUALIZATION
st.subheader(" Cluster Visualization")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df['Bookings_per_Year'], df['Average_Spending'],
    c=df['Cluster'], cmap='viridis', s=100, alpha=0.8, edgecolors='k'
)
plt.xlabel("Bookings per Year")
plt.ylabel("Average Spending (₹)")
plt.title("Customer Segmentation Visualization")
st.pyplot(fig)

#  INSIGHTS 
st.subheader(" Insights")
st.markdown("""
- **Cluster 0:** High-spending and frequent customers — loyal users.  
- **Cluster 1:** Festival-only or occasional customers.  
- **Cluster 2:** New or low-engagement customers.  

These segments can help Devpunya target **marketing campaigns**, offer **personalized services**, 
and improve **customer retention**.
""")

#  DOWNLOAD RESULTS 
st.subheader(" Download Clustered Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Segmented Data (CSV)",
    data=csv,
    file_name="Devpunya_Segmented_Output.csv",
    mime="text/csv"
)

st.success(" Analysis completed successfully!")
