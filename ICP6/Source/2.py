from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#importing the dat set
df= pd.read_csv('weatherHistory.csv')

#searching for attributes which have null values
print(df["Precip Type"].isnull().any())



#dropping the columns
df=df.drop(columns=['Summary','Precip Type','Daily Summary' ],axis=1)

#replacing the null values with mean
df.select_dtypes(include=[np.number]).interpolate().dropna()

X_train,X_test = train_test_split(df,test_size=0.2)

y_train=X_train['Temperature (C)']

X_train=X_train.drop(columns=['Temperature (C)'])
y_test=X_test['Temperature (C)']

X_test=X_test.drop(columns=['Temperature (C)'])

reg=LinearRegression().fit(X_train,y_train)
predict=reg.predict(X_test)
mean_squared_error = mean_squared_error(y_test, predict)
r2_score = r2_score(y_test,predict)
print("mean squared error is :",mean_squared_error)
print("r2 score",r2_score)