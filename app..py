import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# Load assets
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
df = pd.read_csv('processed_data.csv', index_col=0, parse_dates=True)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((len(predictions), 2))], axis=1))[:, 0]
y_test_actual = scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), 2))], axis=1))[:, 0]

# Streamlit UI
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
st.title("📈 LSTM Stock Price Prediction Dashboard")
st.markdown("**Built with TensorFlow + Streamlit | Features: MA, RSI**")

# Metrics
col1, col2, col3 = st.columns(3)
mse = np.mean((predictions - y_test_actual)**2)
mae = np.mean(np.abs(predictions - y_test_actual))
mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

col1.metric("MSE", f"{mse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("MAPE", f"{mape:.2f}%")

# Plot: Actual vs Predicted
fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test_actual, mode='lines', name='Actual Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))
fig.update_layout(title="Actual vs Predicted Stock Prices", xaxis_title="Time", yaxis_title="Price ($)", height=400)
st.plotly_chart(fig, use_container_width=True)

# Plot: Technical Indicators
st.subheader("📊 Technical Indicators")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig2.add_trace(go.Scatter(x=df.index, y=df['MA_20'], mode='lines', name='MA (20)', line=dict(dash='dot')))
fig2.update_layout(title="Close Price + Moving Average", xaxis_title="Date", yaxis_title="Price ($)", height=300)
st.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig3.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI", height=300)
st.plotly_chart(fig3, use_container_width=True)

st.success("🔥 Model successfully deployed!")
