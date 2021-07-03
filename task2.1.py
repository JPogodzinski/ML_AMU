import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv("data2.csv")
print(data)
row1=data.iloc[:,[2]]
row2=data.iloc[:,[3]]

plt.plot(row1, row2, 'ro')

plt.show()