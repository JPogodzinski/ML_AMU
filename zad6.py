from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


data=pd.read_csv('data6.tsv', header=None, sep='\t')
x=np.array(data[0])
x=x.reshape(-1,1)
min_max_scaler = MinMaxScaler()
x=min_max_scaler.fit_transform(x)
y=data[1]
poly=PolynomialFeatures(2, include_bias=False)
x2=poly.fit_transform(x)
poly5=PolynomialFeatures(5, include_bias=False)
x5=poly5.fit_transform(x)

f=LinearRegression()
f.fit(x,y)
f2=LinearRegression()
f2.fit(x2,y)
f5=LinearRegression()
f5.fit(x5,y)
f5R=Ridge()
f5R.fit(x5,y)


fig=plt.figure()
plt.plot(x, y, 'ro')
x=np.arange(-0.1,1,0.01)
ax=fig.add_subplot(1,1,1)
y1=f.coef_*x+f.intercept_
y2=f2.coef_[1]*x**2+f2.coef_[0]*x+f2.intercept_
y5=f5.coef_[4]*x**5+f5.coef_[3]*x**4+f5.coef_[2]*x**3+f5.coef_[1]*x**2+f5.coef_[0]*x+f5.intercept_
y5R=f5R.coef_[4]*x**5+f5R.coef_[3]*x**4+f5R.coef_[2]*x**3+f5R.coef_[1]*x**2+f5R.coef_[0]*x+f5R.intercept_
ax.plot(x, y1, color='yellow', label='first degree polynomial (without regularization)')
ax.plot(x,y2, color='blue', label='second degree polynomial (without regularization)')
ax.plot(x,y5, color='green', label='fifth degree polynomial (without regularization)')
ax.plot(x,y5R, color='black', label='fifth degree polynomial (with regularization')
leg = ax.legend();
plt.show()