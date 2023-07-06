import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def calculate_rsi(data, period):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (delta.where(delta < 0, 0).abs()).fillna(0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Download data from Yahoo Finance
start_date = '2015-01-01'
end_date = '2023-04-29'
ticker = 'BTC-USD'

data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(inplace=True)
data.set_index('Date', inplace=True)

# Calculate simple moving averages
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()

# Calculate exponential moving averages
data['EMA_10'] = data['Close'].ewm(span=10).mean()
data['EMA_30'] = data['Close'].ewm(span=30).mean()

# Calculate the standard deviation
data['STD_10'] = data['Close'].rolling(window=10).std()

# Calculate the daily return
data['Return'] = data['Close'].pct_change()

# Calculate RSI
data['RSI'] = calculate_rsi(data['Close'], 14)

# Drop rows with NaN values
data.dropna(inplace=True)

# Select the features to be used
features = ['Close', 'Volume', 'SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'STD_10', 'Return', 'RSI']
data = data[features]

# Scale the data
feature_scaler = MinMaxScaler()
scaled_data = feature_scaler.fit_transform(data)

# Create a target scaler for closing price
target_scaler = MinMaxScaler()
target_scaler.fit(data['Close'].values.reshape(-1, 1))

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# Prepare the dataset for training
def create_dataset(data, look_back=120):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 120
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Build the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True),input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.2))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32)

predicted_prices = model.predict(X_test)

predicted_prices = target_scaler.inverse_transform(predicted_prices)
actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
percentage_accuracy = np.mean(np.abs(predicted_prices - actual_prices) / actual_prices) * 100

print("Predicted closing price for the next day:", predicted_prices[-1][-1])
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Percentage Accuracy:", percentage_accuracy)

def trade_v2(actual_prices, predicted_prices, initial_balance=5000):
    balance = initial_balance
    position = 0
    entry_price = 0
    balance_list = [initial_balance]
    
    # Calculate the predicted percentage change
    predicted_pct_change = np.diff(predicted_prices, axis=0) / predicted_prices[:-1]

    for i in range(len(actual_prices) - 1):  # We'll stop at the second-to-last price, as we have one less percentage change value
        if position == 0:
            if predicted_pct_change[i] > 0:
                # Buy
                position = 1
                entry_price = actual_prices[i]
            elif predicted_pct_change[i] < 0:
                # Sell
                position = -1
                entry_price = actual_prices[i]
        elif position == 1:
            if predicted_pct_change[i] < 0:
                # Close the long position and sell
                balance += (actual_prices[i] - entry_price) * (balance / entry_price)
                position = 0
        elif position == -1:
            if predicted_pct_change[i] > 0:
                # Close the short position and buy
                balance += (entry_price - actual_prices[i]) * (balance / entry_price)
                position = 0

        # Update the balance based on the current position
        if position == 1:
            current_balance = balance + (actual_prices[i] - entry_price) * (balance / entry_price)
        elif position == -1:
            current_balance = balance + (entry_price - actual_prices[i]) * (balance / entry_price)
        else:
            current_balance = balance

        balance_list.append(current_balance)

    return balance_list

def plot_strategy_vs_buy_and_hold(actual_prices, strategy_balance, initial_balance=5000):
    buy_hold_balance = [initial_balance]
    for i in range(len(actual_prices) - 1):
        buy_hold_balance.append(buy_hold_balance[-1] * (actual_prices[i + 1] / actual_prices[i]))

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_balance, label='Strategy Balance')
    plt.plot(buy_hold_balance, label='Buy and Hold Balance')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.title('Strategy vs Buy and Hold')
    plt.legend()
    plt.show()

strategy_balance_v2 = trade_v2(actual_prices.flatten(), predicted_prices.flatten(), initial_balance=5000)
plot_strategy_vs_buy_and_hold(actual_prices.flatten(), strategy_balance_v2, initial_balance=5000)
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('BTC-USD Actual vs Predicted Prices')
plt.legend()
plt.show()
