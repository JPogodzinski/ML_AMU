from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

alldata=pd.read_csv('flats.tsv', sep='\t', usecols=['cena','Powierzchnia w m2','Liczba pokoi','Liczba pięter w budynku','Piętro','Typ zabudowy','Materiał budynku','Rok budowy','opis'])
alldata['Piętro']=alldata['Piętro'].astype(str)
alldata['Piętro'] = alldata['Piętro'].apply(lambda x: ' 0' if x.strip() in ['parter', 'niski parter'] else x)
alldata['Piętro'] = alldata['Piętro'].apply(pd.to_numeric, errors='coerce')
alldata=pd.get_dummies(alldata, columns=['Typ zabudowy', 'Materiał budynku'])
alldata['opis']=alldata['opis'].astype(str)
alldata=alldata.fillna(alldata.mean())

alldata.drop('opis', inplace=True, axis=1)

split_point = int(0.8 * len(alldata))
data_train = alldata[:split_point]
data_test = alldata[split_point:]


y_train = pd.DataFrame(data_train['cena'])
data_train.drop('cena', inplace=True, axis=1)
x_train = pd.DataFrame(data_train)

model = LinearRegression()
model.fit(x_train, y_train)

y_expected = pd.DataFrame(data_test['cena'])
data_test.drop('cena', inplace=True, axis=1)
x_test = pd.DataFrame(data_test)
y_predicted = model.predict(x_test)


error = mean_squared_error(y_expected, y_predicted)



data=alldata
median=data['cena'].median()
indexNames = data[data['cena'] >5*median].index
data.drop(indexNames, inplace=True)
indexNames = data[data['cena'] <(median/5)].index
data.drop(indexNames, inplace=True)


median=data['Powierzchnia w m2'].median()
indexNames = data[data['Powierzchnia w m2'] >5*median].index
data.drop(indexNames, inplace=True)
indexNames = data[data['Powierzchnia w m2'] <(median/5)].index
data.drop(indexNames, inplace=True)


split_point2 = int(0.8 * len(data))
data_train2 = data[:split_point2]
data_test2 = data[split_point2:]


y_train2 = pd.DataFrame(data_train2['cena'])
data_train2.drop('cena', inplace=True, axis=1)
x_train2 = pd.DataFrame(data_train2)

model2 = LinearRegression()
model2.fit(x_train2, y_train2)

y_expected2 = pd.DataFrame(data_test2['cena'])
data_test2.drop('cena', inplace=True, axis=1)
x_test2 = pd.DataFrame(data_test2)
y_predicted2 = model2.predict(x_test2)
error2 = mean_squared_error(y_expected2, y_predicted2)


print(f"MSE for first model is {error}")
print(model.score(x_test, y_expected))
print(f"MSE for second model is {error2}")
print(model2.score(x_test2, y_expected2))

