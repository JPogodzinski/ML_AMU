from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression



data=pd.read_csv('fires_thefts.csv', header=None, sep=',')
print(data)

split_point = int(0.8 * len(data))
data_train = data[:split_point]
data_test = data[split_point:]

# Learning model
y_train = pd.DataFrame(data_train[1])
x_train = pd.DataFrame(data_train[0])
model = LinearRegression()
model.fit(x_train, y_train)

y_expected = pd.DataFrame(data_test[1])
x_test = pd.DataFrame(data_test[0])
y_predicted = model.predict(x_test)

print(y_predicted)

error = mean_squared_error(y_expected, y_predicted)

print(f"MSE is {error}")
print(model.score(x_test, y_expected))
