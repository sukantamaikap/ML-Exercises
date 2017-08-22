import pandas as pd
import quandl

df = quandl.get('WIKI/TSLA')
print(df.head())
print(df.shape)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) * 100
df['PCT_Change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', "Adj. Volume"]]
print(df.head())
print(df.shape)