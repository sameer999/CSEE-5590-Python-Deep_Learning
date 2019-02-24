import pandas as pd

#importing the dataset
titanic = pd.read_csv('titanic.csv')

print(titanic['Sex'])

#converting categorical data to  numeric data by using map function
titanic['Sex'] = titanic['Sex'].map( {'female': 1, 'male': 0} )

print("\nafter coonverting categorical values to numeric\n",titanic['Sex'])

#finding the correlation
print("\ncorrelation for sex and survived: ", titanic['Sex'].corr(titanic['Survived']))

print('\nstandard correlation(pearson method):\n ', titanic.corr())
