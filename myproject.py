import pandas as pd 
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit Page Config
st.set_page_config(page_title="User Clustering App", layout="wide")

# Load Dataset
df_original = pd.read_csv("user_behavior_dataset.csv")
df = df_original.copy()

# Drop User ID column
df.drop(['User ID','User Behavior Class'], axis=1, inplace=True)
df_original.drop('User ID', axis=1, inplace=True)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 
                      'Battery Drain (mAh/day)', 'Number of Apps Installed', 
                      'Data Usage (MB/day)', 'Age']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Train K-Means Model with Optimal Clusters
final_model = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = final_model.fit_predict(df[numerical_features])

# Assign Remarks Based on Cluster
df.loc[df.Cluster == 0, 'Remarks'] = 'Light User'
df.loc[df.Cluster == 1, 'Remarks'] = 'Moderate User'
df.loc[df.Cluster == 2, 'Remarks'] = 'Heavy User'

# Save model and scaler
joblib.dump(final_model, "clustering_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Prepare Data for Display
df_display = df_original.copy()
df_display['Cluster'] = df['Cluster']
df_display['Remarks'] = df['Remarks']

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Clustering Analysis", "Predict Your Cluster"])

# Home Page
if page == "Home":
    st.title("📊 Clustering Mobile App Users")
    st.markdown("### Welcome to the User Clustering App!")
    st.markdown("This app segments mobile users based on their activity data using machine learning.")
    
    # Display Background Image using Streamlit
    st.image("Active-users.jpeg", use_container_width=True)

# Clustering Analysis Page
elif page == "Clustering Analysis":
    st.title("📌 Clustering Analysis")
    
    # Show clustered data sample
    st.subheader("📌 Clustered Data Sample")
    st.dataframe(df_display.head(10))

    # Show Cluster Distribution
    st.subheader("📊 Cluster Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Cluster"], palette="viridis", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Users")
    st.pyplot(fig)

    # Additional Informative Graphs
    st.subheader("📊 Key Feature Distributions")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    features = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)', 'Battery Drain (mAh/day)']
    colors = ["blue", "green", "red", "purple"]
    for i, feature in enumerate(features):
        sns.histplot(df_original[feature], bins=30, kde=True, ax=axes[i // 2, i % 2], color=colors[i])
        axes[i // 2, i % 2].set_title(f"{feature} Distribution")
    st.pyplot(fig)

# Predict Your Cluster Page
elif page == "Predict Your Cluster":
    st.title("🔮 Predict Your Cluster")

    # Load saved model and scaler
    final_model = joblib.load("clustering_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Collect user input
    st.subheader("Enter Your Details:")
    user_input = {}
    
    for feature in numerical_features:
        if feature == 'Age' or feature == 'Number of Apps Installed':
            user_input[feature] = st.number_input(f"{feature}", min_value=int(df_display[feature].min()), max_value=int(df_display[feature].max()), value=int(df_display[feature].mean()), step=1)
        else:
            user_input[feature] = st.slider(f"{feature}", min_value=float(df_display[feature].min()), max_value=float(df_display[feature].max()), value=float(df_display[feature].mean()))

    if st.button("Predict My Cluster"):
        input_df = pd.DataFrame([user_input])
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        predicted_cluster = final_model.predict(input_df)[0]
        predicted_remark = "Light User" if predicted_cluster == 0 else "Moderate User" if predicted_cluster == 1 else "Heavy User"
        st.success(f"🎯 Your predicted cluster: {predicted_cluster} ({predicted_remark})")
