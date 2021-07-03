from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

alldata=pd.read_csv('gratkapl-centrenrm.csv', header=0, sep=',')
print(alldata)

print('Number of records before deletion of NaN:', len(alldata))

alldata = alldata.dropna()  # usunięcie rekordów zawierających NaN

print('Number of records after deletion of NaN:', len(alldata))

median=alldata['Price'].median()
print(median)
indexNames = alldata[alldata['Price'] >5*median].index
alldata.drop(indexNames, inplace=True)
indexNames = alldata[alldata['Price'] <(median/5)].index
alldata.drop(indexNames, inplace=True)
print('Number of records after removing extremes in price:', len(alldata))

median=alldata['SqrMeters'].median()
print(median)
indexNames = alldata[alldata['SqrMeters'] >5*median].index
alldata.drop(indexNames, inplace=True)
indexNames = alldata[alldata['SqrMeters'] <(median/5)].index
alldata.drop(indexNames, inplace=True)
print('Number of records after removing extremes in size:', len(alldata))

split_point = int(0.8 * len(alldata))
data_train = alldata[:split_point]
data_test = alldata[split_point:]

FEATURES=['Price','Rooms','SqrMeters','Floor']

y_train = pd.DataFrame(data_train['Centre'])
x_train = pd.DataFrame(data_train[FEATURES])
model = LogisticRegression()
model.fit(x_train, y_train)

y_expected = pd.DataFrame(data_test['Centre'])
x_test = pd.DataFrame(data_test[FEATURES])
y_predicted = model.predict(x_test)

print(y_predicted)

precision, recall, fscore, support = precision_recall_fscore_support(y_expected, y_predicted, average="micro")

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-score: {fscore}")

score = model.score(x_test, y_expected)

print(f"Model score: {score}")