import pandas as pd

train_df = pd.read_csv('titanic.csv')

train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} )


print(type(train_df['Sex'][1]))
print(train_df['Survived'].corr(train_df['Sex']))