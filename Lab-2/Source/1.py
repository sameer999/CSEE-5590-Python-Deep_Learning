import pandas as pd
import math as ma
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

#loading the dataset
df=pd.read_csv('patient.csv')

#finding columns with null attributes
print(df.isnull().sum())

#loading the values of comfort column
c=df['COMFORT']

#finding the mean for the attribute comfort
m=c.mean(skipna=True)
m=ma.ceil(m)

#replacing the null values with mean of Comfort values
df['COMFORT']=df['COMFORT'].fillna(m)

#Training set : only feature attributes
X=df.drop('COMFORT',axis=1)

#converting the categorical to numeric attributes
X=pd.get_dummies(X,columns=["L-CORE","L-SURF","L-O2","L-BP","SURF-STBL","CORE-STBL","BP-STBL","DECISION"])

#test set: target attribute valaues
y=df.iloc[:, 7].values

#Splitting the data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

#Scaling the values
from sklearn.preprocessing import StandardScaler
scalar_X = StandardScaler()
X_train = scalar_X.fit_transform(X_train)
X_test = scalar_X.transform(X_test)

#Naive Bayes
model=GaussianNB()
model.fit(X_train,y_train)
y_p1=model.predict(X_test)
acc1=metrics.accuracy_score(y_test,y_p1)
print("Gaussian Naive Bayes accuracy :",acc1)

#SVM
clf=SVC(kernel='linear',C=1).fit(X_train,y_train)
y_p2=clf.predict(X_test)
acc2 = metrics.accuracy_score(y_test, y_p2)
print("svm accuracy :", acc2)

#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_p3 = knn.predict(X_test)
acc3 = metrics.accuracy_score(y_test, y_p3)
print("KNN accuracy :",acc3)