import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('train.csv', sep=',',usecols=(62,80))
#print(df)
#print(df.info())

#print(type(df))
y = df['SalePrice']
x = df['GarageArea']
print('original shape of datashape',df.shape)
plt.scatter(x,y)
plt.title("original dataframe")
plt.ylabel("SalesPrice")
plt.xlabel("GarageArea")
plt.show()

z = np.abs(stats.zscore(df))
threshold = 3
print(np.where(z > 3))
modified_df = df[(z < 3).all(axis=1)]

y = modified_df['SalePrice']
x = modified_df['GarageArea']
print('after removing outliers',modified_df.shape)

plt.scatter(x,y)
plt.title("after deleting outliers")
plt.ylabel("SalesPrice")
plt.xlabel("GarageArea")
plt.show()