"""
msft_lstm_forecast_v3.py
---------------------------------
Microsoft (MSFT) stock price forecasting using a Bidirectional LSTM model.
✅ Uses saved scaler (joblib)
✅ Adds Technical Indicators (EMA20, EMA50, RSI, MACD)
✅ Predicts UP / DOWN classification & prints scikit-learn classification report
"""

# ============================================================
# 1️⃣ Imports
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

### scaler saving
import joblib

# ============================================================
# 2️⃣ Download and prepare data
# ============================================================
ticker = "MSFT"
data = yf.download(ticker, start="2016-01-01", end="2025-11-06")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# ============================================================
# ✅ Add Technical Indicators
# ============================================================
data["EMA20"] = data["Close"].ewm(span=20).mean()
data["EMA50"] = data["Close"].ewm(span=50).mean()
data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(14).mean()))
data["MACD"] = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()

# Select Close & indicators
data = data[["Close", "EMA20", "EMA50", "RSI", "MACD"]].copy()
data.dropna(inplace=True)

# ============================================================
# ✅ Save scaler
# ============================================================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
joblib.dump(scaler, "scaler.pkl")  
print("✅ Scaler saved to scaler.pkl")

data_scaled = pd.DataFrame(scaled, index=data.index, columns=data.columns)

print(f"✅ Data loaded and scaled. Shape: {data_scaled.shape}")
print(data_scaled.head())

# ============================================================
# 3️⃣ Create Windowed Dataset
# ============================================================
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=60):
    def str_to_datetime(s): return datetime.datetime.strptime(s, "%Y-%m-%d")

    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date
    dates, X, Y = [], [], []

    all_dates = dataframe.index.to_list()
    i = all_dates.index(target_date)

    while True:
        if i - n < 0: break

        window = dataframe.iloc[i - n:i + 1].to_numpy()   # (61, 5 features)
        if len(window) != n + 1: break

        x = window[:-1]
        y = window[-1][0]  # predict CLOSE only

        dates.append(all_dates[i])
        X.append(x)
        Y.append(y)

        if all_dates[i] == last_date: break
        i += 1
        if i >= len(all_dates): break

    return pd.DataFrame({"Target Date": dates, "X": X, "Y": Y})

windowed_df = df_to_windowed_df(data_scaled, "2017-03-01", "2025-11-05", n=60)
print(f"✅ Windowed dataset created: {windowed_df.shape}")

# ============================================================
# 4️⃣ Convert to X, y
# ============================================================
dates = windowed_df["Target Date"].to_numpy()
X = np.stack(windowed_df["X"].values)
y = np.stack(windowed_df["Y"].values).astype(np.float32)

# ============================================================
# 5️⃣ Time-based Split
# ============================================================
dates = pd.to_datetime(dates)

train = dates < "2022-01-01"
val = (dates >= "2022-01-01") & (dates < "2024-01-01")
test = dates >= "2024-01-01"

X_train, y_train = X[train], y[train]
X_val, y_val = X[val], y[val]
X_test, y_test = X[test], y[test]

print(f"✅ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ============================================================
# 6️⃣ Build Model
# ============================================================
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(60, X.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(128)),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1)
])

model.compile(loss="mse", optimizer=Adam(learning_rate=0.0005), metrics=["mean_absolute_error"])
model.summary()

# ============================================================
# 7️⃣ Train
# ============================================================
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save("msft_bilstm_60day.h5")
print("✅ Model saved as msft_bilstm_60day.h5")

# ============================================================
# 8️⃣ Evaluate (Regression Metrics)
# ============================================================
loss, mae = model.evaluate(X_test, y_test)
print(f"\n✅ Test MSE: {loss:.6f} | Test MAE: {mae:.6f}")

y_pred = model.predict(X_test)

scaler = joblib.load("scaler.pkl") 
y_pred_rescaled = scaler.inverse_transform(
    np.hstack([y_pred, np.zeros((len(y_pred), 4))])
)[:, 0]

y_test_rescaled = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 4))])
)[:, 0]

rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"✅ RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# ============================================================
# Classification (Up/Down Prediction)
# ============================================================
y_test_updown = np.where(np.diff(y_test_rescaled, prepend=y_test_rescaled[0]) > 0, 1, 0)
y_pred_updown = np.where(np.diff(y_pred_rescaled, prepend=y_pred_rescaled[0]) > 0, 1, 0)

print("\n📊 UP / DOWN Classification Report:")
print(classification_report(y_test_updown, y_pred_updown, target_names=["DOWN", "UP"]))

# ============================================================
# 🔟 Visualization
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(dates[test], y_test_rescaled, label="Actual")
plt.plot(dates[test], y_pred_rescaled, label="Predicted")
plt.title("MSFT Close Price Prediction (Bidirectional LSTM, 60-day Window + Indicators)")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid()
plt.show()
