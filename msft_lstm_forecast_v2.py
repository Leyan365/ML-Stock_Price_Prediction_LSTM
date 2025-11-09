"""
msft_lstm_forecast_v2.py
---------------------------------
Microsoft (MSFT) stock price forecasting using a Bidirectional LSTM model.
✅ Optimized for 2016–2025 data
✅ 60-day lookback window
✅ MinMax scaling
✅ Bidirectional LSTM + Dropout + BatchNormalization
✅ Learning rate scheduling & early stopping
✅ Robust date-based train/val/test split
✅ Actual vs Predicted visualization
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================
# 2️⃣ Download and prepare data
# ============================================================
ticker = "MSFT"
data = yf.download(ticker, start="2016-01-01", end="2025-11-06")

# Drop multi-level columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Select Close price and clean
data = data[['Close']].copy()
data.dropna(inplace=True)

# Scale the data
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])

print(f"✅ Data loaded. Shape: {data.shape}")
print(data.head())

# ============================================================
# 3️⃣ Create windowed dataset
# ============================================================
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=60):
    """Convert a time series into a supervised windowed dataset."""
    def str_to_datetime(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d")

    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date
    dates, X, Y = [], [], []

    all_dates = dataframe.index.to_list()
    i = all_dates.index(target_date)

    while True:
        if i - n < 0:
            break
        window = dataframe.iloc[i - n:i + 1]['Close'].to_numpy()
        if len(window) != n + 1:
            break

        x = window[:-1]
        y = window[-1]
        dates.append(all_dates[i])
        X.append(x)
        Y.append(y)

        if all_dates[i] == last_date:
            break
        i += 1
        if i >= len(all_dates):
            break

    X = np.array(X)
    windowed_df = pd.DataFrame({'Target Date': dates})
    for j in range(n):
        windowed_df[f'Target-{n-j}'] = X[:, j]
    windowed_df['Target'] = Y

    return windowed_df

# Build windowed dataframe (ensure we have 60 days before first target)
windowed_df = df_to_windowed_df(data, '2017-03-01', '2025-11-05', n=60)
print(f"✅ Windowed dataframe created: {windowed_df.shape}")
print(windowed_df.head())

# ============================================================
# 4️⃣ Convert to X, y, dates
# ============================================================
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

# ============================================================
# 5️⃣ Time-based train/val/test split
# ============================================================
split_date_1 = '2022-01-01'
split_date_2 = '2024-01-01'

train_mask = dates < split_date_1
val_mask = (dates >= split_date_1) & (dates < split_date_2)
test_mask = dates >= split_date_2

dates_train, X_train, y_train = dates[train_mask], X[train_mask], y[train_mask]
dates_val, X_val, y_val = dates[val_mask], X[val_mask], y[val_mask]
dates_test, X_test, y_test = dates[test_mask], X[test_mask], y[test_mask]

print(f"✅ Split complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ============================================================
# 6️⃣ Visualize splits
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(dates_train, y_train, label="Train")
plt.plot(dates_val, y_val, label="Validation")
plt.plot(dates_test, y_test, label="Test")
plt.title("Train / Validation / Test Split")
plt.legend()
plt.show()

# ============================================================
# 7️⃣ Build Bidirectional LSTM model
# ============================================================
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(60, 1)),
    Dropout(0.3),
    Bidirectional(LSTM(128)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005), metrics=['mean_absolute_error'])
model.summary()

# ============================================================
# 8️⃣ Callbacks
# ============================================================
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ============================================================
# 9️⃣ Train model
# ============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# ============================================================
# 🔟 Evaluate on test set
# ============================================================
loss, mae = model.evaluate(X_test, y_test)
print(f"\n✅ Test MSE: {loss:.6f} | Test MAE: {mae:.6f}")

# ============================================================
# 11️⃣ Predict and visualize
# ============================================================
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae_rescaled = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"✅ RMSE: {rmse:.4f} | MAE: {mae_rescaled:.4f}")

# Plot predictions
plt.figure(figsize=(12,6))
plt.plot(dates_test, y_test_rescaled, label="Actual", color='blue')
plt.plot(dates_test, y_pred_rescaled, label="Predicted", color='red')
plt.title("MSFT Close Price Prediction (Bidirectional LSTM, 60-day Window)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
