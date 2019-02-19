from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#Importing the dataset 'iris'
iris = datasets.load_iris()
#print(iris)

#splitting data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2,random_state=20)

#creating the classifier
lc = svm.SVC(kernel="linear")

#training the classifier
lc.fit(x_train, y_train)

#Prediction
y_predict = lc.predict(x_test)


#Evaluation
print("accuracy", metrics.accuracy_score(y_test, y_predict))
print("classification_report\n",metrics.classification_report(y_test,y_predict))
print("confusion matrix\n",metrics.confusion_matrix(y_test,y_predict))



