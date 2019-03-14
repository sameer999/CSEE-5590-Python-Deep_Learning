import pandas as pd
wine = pd.read_csv('wine.csv')
#print(wine)

#finding the correlation between all columns
c= wine.corr()
print(c[c>0.0])