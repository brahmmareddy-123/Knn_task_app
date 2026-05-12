import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="KNN Regression App", layout="centered")

st.title("📈 KNN Regression using Streamlit")
st.write("This app performs KNN Regression using the Iris dataset.")

# Load dataset
df = pd.read_csv("Iris.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm']]
y = df['PetalWidthCm']

# Sidebar settings
st.sidebar.header("Model Settings")

n_neighbors = st.sidebar.slider(
    "Number of Neighbors (K)",
    min_value=1,
    max_value=15,
    value=5
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2
)

weights = st.sidebar.selectbox(
    "Weights",
    ['uniform', 'distance']
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# Create model
model = KNeighborsRegressor(
    n_neighbors=n_neighbors,
    weights=weights
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")

st.write(f"**Mean Absolute Error:** {mae:.4f}")
st.write(f"**Mean Squared Error:** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# Prediction section
st.subheader("Predict Petal Width")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    value=1.4
)

if st.button("Predict"):
    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length
    ]])

    st.success(
        f"Predicted Petal Width: {prediction[0]:.2f} cm"
    )
