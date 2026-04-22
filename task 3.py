import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
print("Libraries loaded successfully")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

df = df.asfreq('MS')

print("\nData Preview:")
print(df.head())

plt.figure(figsize=(10,5))
plt.plot(df['Passengers'])
plt.title("Air Passengers Data")
plt.show()

decomposition = seasonal_decompose(df, model='multiplicative')

fig = decomposition.plot()
fig.set_size_inches(10,8)
plt.show()


df['MA_6'] = df['Passengers'].rolling(window=6).mean()
df['MA_12'] = df['Passengers'].rolling(window=12).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Passengers'], label='Actual')
plt.plot(df['MA_6'], label='MA 6')
plt.plot(df['MA_12'], label='MA 12')
plt.legend()
plt.title("Moving Average")
plt.show()

train = df.iloc[:-12]
test = df.iloc[-12:]

model = ARIMA(train['Passengers'], order=(2,1,1), seasonal_order=(1,1,1,12))
result = model.fit()

forecast = result.forecast(steps=12)

rmse = np.sqrt(mean_squared_error(test['Passengers'], forecast))
print("\nRMSE:", rmse)

plt.figure(figsize=(10,5))
plt.plot(train.index, train['Passengers'], label='Train')
plt.plot(test.index, test['Passengers'], label='Actual')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.title("Forecast vs Actual")
plt.show()