from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd

#Importing the dataset 'iris'
iris=pd.read_csv("Iris.csv")


#Preprocessing data
a=iris.drop('class',axis=1)
b=iris['class']

#splitting data into training data and testing data
x_train,x_test,y_train,y_test=model_selection.train_test_split(a,b,test_size=0.2,random_state=20)

#creating and Training the classifier
model=GaussianNB()
model.fit(x_train,y_train)

#Prediction
y_pred=model.predict(x_test)

#Evaluation

print("accuracy score:",metrics.accuracy_score(y_test,y_pred))
print("classification_report\n",metrics.classification_report(y_test,y_pred))
print("confusion matrix\n",metrics.confusion_matrix(y_test,y_pred))