 
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.set_page_config(page_title="NSE Nifty Stock Prediction", layout="centered")

# -------------------------
# 1. Website Title
# -------------------------
st.title("📈 NSE Nifty Stock Prediction")
st.write("Select a company to predict the next day's closing price using LSTM.")

# -------------------------
# 2. Full Nifty 50 Companies
# -------------------------
nifty_50 = {
    'ADANIENT.NS': 'Adani Enterprises', 'ADANIPORTS.NS': 'Adani Ports',
    'ASIANPAINT.NS': 'Asian Paints', 'AXISBANK.NS': 'Axis Bank',
    'BAJAJ-AUTO.NS': 'Bajaj Auto', 'BAJFINANCE.NS': 'Bajaj Finance',
    'BAJAJFINSV.NS': 'Bajaj Finserv', 'BHARTIARTL.NS': 'Bharti Airtel',
    'BPCL.NS': 'BPCL', 'BRITANNIA.NS': 'Britannia', 'CIPLA.NS': 'Cipla',
    'COALINDIA.NS': 'Coal India', 'DIVISLAB.NS': 'Divi’s Labs',
    'DRREDDY.NS': 'Dr. Reddy’s Labs', 'EICHERMOT.NS': 'Eicher Motors',
    'GRASIM.NS': 'Grasim Industries', 'HCLTECH.NS': 'HCL Tech',
    'HDFCBANK.NS': 'HDFC Bank', 'HDFCLIFE.NS': 'HDFC Life',
    'HEROMOTOCO.NS': 'Hero MotoCorp', 'HINDALCO.NS': 'Hindalco',
    'HINDUNILVR.NS': 'Hindustan Unilever', 'ICICIBANK.NS': 'ICICI Bank',
    'INDUSINDBK.NS': 'IndusInd Bank', 'INFY.NS': 'Infosys', 'ITC.NS': 'ITC',
    'JSWSTEEL.NS': 'JSW Steel', 'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro', 'M&M.NS': 'Mahindra & Mahindra',
    'MARUTI.NS': 'Maruti Suzuki', 'NESTLEIND.NS': 'Nestle India',
    'NTPC.NS': 'NTPC', 'ONGC.NS': 'ONGC', 'POWERGRID.NS': 'Power Grid',
    'RELIANCE.NS': 'Reliance Industries', 'SBILIFE.NS': 'SBI Life',
    'SBIN.NS': 'SBI', 'SHREECEM.NS': 'Shree Cement', 'SUNPHARMA.NS': 'Sun Pharma',
    'TATACONSUM.NS': 'Tata Consumer', 'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel', 'TCS.NS': 'TCS', 'TECHM.NS': 'Tech Mahindra',
    'TITAN.NS': 'Titan Company', 'ULTRACEMCO.NS': 'UltraTech Cement',
    'UPL.NS': 'UPL', 'WIPRO.NS': 'Wipro'
}

symbol = st.selectbox("Choose a Company", list(nifty_50.keys()), format_func=lambda x: nifty_50[x])

# -------------------------
# 3. Fetch Stock Data
# -------------------------
@st.cache_data
def get_data(symbol):
    return yf.download(symbol, start="2015-01-01", end="2024-12-31")

if symbol:
    df = get_data(symbol)
    st.subheader(f"{nifty_50[symbol]} - Last 5 Days")
    st.dataframe(df.tail())

    # -------------------------
    # 4. Preprocess
    # -------------------------
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    train_size = int(0.7 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # -------------------------
    # 5. LSTM Model
    # -------------------------
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # -------------------------
    # 6. Predict Next Day
    # -------------------------
    last_60 = scaled_data[-60:]
    X_input = last_60.reshape(1, 60, 1)
    pred_scaled = model.predict(X_input)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    st.success(f"🔮 Predicted Next Day Closing Price: ₹{pred_price:.2f}")

    # -------------------------
    # 7. Plot Graph
    # -------------------------
    train_data = data[:train_size + sequence_length]
    test_data = data[train_size + sequence_length:]
    predicted = scaler.inverse_transform(model.predict(X_test))

    fig, ax = plt.subplots()
    ax.plot(train_data, label='Train')
    ax.plot(test_data, label='Actual', color='orange')
    ax.plot(range(len(train_data), len(train_data)+len(predicted)), predicted, label='Predicted', color='green')
    ax.set_title(f"{nifty_50[symbol]} - Price Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Closing Price (₹)")
    ax.legend()
    st.pyplot(fig)
