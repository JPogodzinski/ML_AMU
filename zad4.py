import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


alldata=pd.read_csv('gratkapl-centrenrm.csv', header=0, sep=',')
print(alldata)

print('Number of records before deletion of NaN:', len(alldata))

alldata = alldata.dropna()  # usunięcie rekordów zawierających NaN

print('Number of records after deletion  NaN:', len(alldata))

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

print('The effectiveness for the test set is ',  model.score(x_test, y_expected) * 100, '%')

# Random classifier
y = np.random.choice([0, 1], size=(403, ))
print('Matrix Y: \n', y)
print('The effectiveness for the test set is (compared to a random classifier) ', model.score(x_test, y) * 100, '%')