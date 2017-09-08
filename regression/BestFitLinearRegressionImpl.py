from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

#
def generate_best_fit(x, y):
    m = ((mean(x) * mean(y)) - mean(x*y))/(mean(x)*mean(x) - mean(x*x))
    print("m found : ", m)
    b = mean(y) - m*mean(x)
    print('b found : ', b)
    return m, b


def predict_y(x):
    y = (m*x)+b
    print('for x : ',x)
    print('y prediction is : ', y)
    return y

xs = np.random.randint(22, size=6)
ys = np.random.randint(12, size=6)
print('xs : ', xs)
print('ys : ', ys)

m, b = generate_best_fit(xs, ys)
regression_line = [(m*x)+b for x in xs]
print(regression_line)
x_predict = 22
y_predict = predict_y(x_predict)

plt.scatter(xs, ys)
plt.scatter(x_predict, y_predict)
plt.plot(xs, regression_line)
plt.show()