import pandas as pd
import quandl
import math

# pull data from Quandl
df = quandl.get('WIKI/TSLA')
print("raw data from Quandl")
print(df.head())
print(df.shape)

# define your features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) * 100
df['PCT_Change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', "Adj. Volume"]]
print('features chosen ....')
print(df.head())
print(df.shape)

# define the label
forecast_column = 'Adj. Close'
df.fillna(-9999, inplace=True)
# define the forecast lenght
forecast_out = int(math.ceil(0.1 * len(df)))
df['label'] = df[forecast_column].shift(-forecast_out)
df.dropna(inplace=True)

print('label chosen ...')
print(df.head())
print(df.shape)
print(df.tail())