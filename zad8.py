from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('mushrooms.tsv', sep='\t', header=None)

data[0] = data[0].apply(lambda x: 0 if x =='e' else 1)
non_dummy=[0]
dummy_cols=list(set(data)-set(non_dummy))
data=pd.get_dummies(data, columns=dummy_cols)

split_point = int(0.8 * len(data))
data_train = data[:split_point]
data_test = data[split_point:]

y_train=data_train[0]
data_train.drop(0,inplace=True,axis=1)
x_train=data_train

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)

model=SGDClassifier(max_iter=1000, tol=1e-3)
model.fit(x_train_scaled,y_train)

y_expected=data_test[0]
data_test.drop(0,inplace=True,axis=1)
x_test=data_test
x_test_scaled=scaler.fit_transform(x_test)

y_predicted=model.predict(x_test_scaled)

error=mean_squared_error(y_expected, y_predicted)

print(error)


