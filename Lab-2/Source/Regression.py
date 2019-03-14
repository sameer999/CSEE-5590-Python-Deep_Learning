import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#importing the dataset
dataset = pd.read_csv('servo.csv')

#categorical to numeric conversion
dataset['motor']=dataset['motor'].map({'A':1,'B':2,'C':3,'D':4,'E':5})
dataset['motor']=dataset['screw'].map({'A':1,'B':2,'C':3,'D':4,'E':5})

#finding correlation
print(dataset.corr())

#choosing the test data
y = dataset.iloc[:,2].values

#train data :dropping some attributes since they are less correlated to predicted attribute
X = dataset.drop(['motor','screw','pgain','class'],axis=1)

#split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 35)

#scaling
from sklearn.preprocessing import StandardScaler
scalar_X = StandardScaler()
X_train = scalar_X.fit_transform(X_train)
X_test = scalar_X.transform(X_test)

#multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)

#predicting the values
y_pred = regressor.predict(X_test)

#evaluation
from sklearn.metrics import mean_squared_error, r2_score
print("Variance score: %.2f" % r2_score(y_test,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

#plot
plt.scatter(y_pred, y_test, alpha=.75,
            color='b')  # alpha helps to show overlapping data
plt.xlabel('Predicted ')
plt.ylabel('Actual ')
plt.title('Multiple Regression Model')
plt.show()
