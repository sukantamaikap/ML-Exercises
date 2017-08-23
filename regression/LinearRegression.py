import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# pull data from Quandl
df = quandl.get('WIKI/TSLA')
print("raw data from Quandl")
print(df.head())
print(df.shape)

# define your features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', "Adj. Volume"]]
print('features chosen ....')
print(df.head())
print(df.shape)

# define the label
forecast_column = 'Adj. Close'
df.fillna(-9999, inplace=True)
# define the forecast lenght
forecast_out = int(math.ceil(0.01 * len(df)))
print("forecast lengthh : " + str(forecast_out))
df['label'] = df[forecast_column].shift(-forecast_out)
df.dropna(inplace=True)

print('label chosen ...')
print(df.head())
print(df.shape)
print(df.tail())

#features
X = np.array(df.drop(['label'], 1))
# X = X[:-forecast_out]
X_lately = X[-forecast_out:]
X = preprocessing.scale(X)
#lable
y = np.array(df['label'])

print(len(X), len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.head())
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
# plt.show()