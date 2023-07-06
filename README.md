# Bitcoin Price Prediction and Trading Strategy

This project uses a deep learning model (Bidirectional LSTM) for predicting the daily closing price of Bitcoin (BTC). It downloads historical data from Yahoo Finance, performs feature engineering, and uses this data to train the LSTM model. The trained model is then used to predict future prices. Additionally, it also includes a simple trading strategy that takes these predictions into account to make trades.

## Dependencies

The script depends on the following Python libraries:

- Numpy
- Pandas
- Matplotlib
- YFinance
- Scikit-Learn
- TensorFlow
- Keras

You can install these dependencies using pip:

```shell
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
