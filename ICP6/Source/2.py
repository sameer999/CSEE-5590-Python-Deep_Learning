from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#importing the dat set
df= pd.read_csv('weatherHistory.csv')

#searching for attributes which have null values
print(df["Precip Type"].isnull().any())

#converting the categorical data  to numeric type
#df = pd.get_dummies(df, columns=["Precip Type","Summary","Daily Summary"])

#finding the correlation for better training of the model by selecting the appropriate features
print(df.corr())

#dropping the columns since they have less correlation with target class
df=df.drop(columns=['Summary','Precip Type','Daily Summary'],axis=1)

#replacing the null values with mean
df.select_dtypes(include=[np.number]).interpolate().dropna()

#splitting into test and train data
X_train,X_test = train_test_split(df,test_size=0.2)

y_train=X_train['Temperature (C)']

X_train=X_train.drop(columns=['Temperature (C)'])
y_test=X_test['Temperature (C)']

X_test=X_test.drop(columns=['Temperature (C)'])

#creation of regression model and training it
reg=LinearRegression().fit(X_train,y_train)

#predicting the target
predict=reg.predict(X_test)

#evaluation of model using metrics
mean_squared_error = mean_squared_error(y_test, predict)
r2_score = r2_score(y_test,predict)
print("mean squared error is :",mean_squared_error)
print("r2 score",r2_score)