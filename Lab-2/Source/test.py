mport pandas as pd
import math as ma
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
data=pd.read_csv('patient.csv')
f1=data['COMFORT']
b=f1.mean(skipna=True)
b=ma.ceil(b)
print(b)
data['COMFORT']=data['COMFORT'].fillna(b)
print(data.isnull().sum())
#Encoding the Categorical data
#data['BP-STBL']=data['BP-STBL'].astype('category')
#print(data.dtypes)
#data['BP-STBL']=data['BP-STBL'].cat.codes
data=pd.get_dummies(data,columns=["BP-STBL"])
print(data)