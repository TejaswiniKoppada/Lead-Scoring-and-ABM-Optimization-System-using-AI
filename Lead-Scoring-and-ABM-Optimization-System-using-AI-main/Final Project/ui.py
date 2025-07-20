import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Try importing plotly and handle missing module gracefully
try:
    import plotly.express as px
    plotly_available = True
except ModuleNotFoundError:
    plotly_available = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
def load_data():
    return pd.read_csv("Lead Scoring.csv")

df = load_data()

st.title("AI Driven Lead Scoring and ABM optimization system")

# Data Overview
st.sidebar.header("Data Overview")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df.head())

# Data Summary
st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")

# Conversion Rate Visualization
st.subheader("Lead Conversion Rate")
fig, ax = plt.subplots()
sns.countplot(x='Converted', data=df, ax=ax)
st.pyplot(fig)

# Feature Importance (Using RandomForest for simplicity)
st.subheader("Feature Importance Analysis")

# Preprocessing

df.dropna(inplace=True)  # Handling missing values

# Convert categorical columns to numeric
for col in df.select_dtypes(include=['object']).columns:
    if col != "Converted":  # Exclude target variable if it's categorical
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["Converted"])  # Features
y = df["Converted"]  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(importances)

# Model Evaluation
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Interactive Scatter Plot
st.subheader("Interactive Data Visualization")
if plotly_available:
    x_axis = st.selectbox("Select X-axis feature", df.columns)
    y_axis = st.selectbox("Select Y-axis feature", df.columns)
    st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, color="Converted"))
else:
    st.warning("Plotly is not installed. Run `pip install plotly` to enable interactive visualizations.")
