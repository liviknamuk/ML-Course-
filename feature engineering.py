import pandas as pd

import numpy  as  np


url ="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


print(url)


df =pd.read_csv(url)


print(df)



print(df.info())


print(df.isnull().sum())

df.columns =df.columns.str.strip()


df.columns =df.columns.str.lower()


print(df.columns)


df =df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]



print(df)


df ['age'] =df['age'].fillna(df['age'].median())

df ['sex'] =df['sex'].map({"male":0 , 'female':1})


print(df)



df ['embarked'] =df['embarked'].fillna('S')

df ['embarked'] =df['embarked'].map({'S':0, 'C':1, 'Q': 2})

print(df)




df ['famaly_size']  =df['sibsp'] + df['parch'] + 1


df ['age_size'] =df['famaly_size'].apply(lambda X:1 if X < 12 else 0)



print(df)



from sklearn.model_selection  import train_test_split


figg=['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked', 'famaly_size', 'age_size']


X =df[figg]

y =df['survived']


print(y)


X_train , X_test , y_train  , y_test =train_test_split(X, y, test_size=0.2)



from sklearn.metrics import accuracy_score


from sklearn.linear_model import LogisticRegression


log =LogisticRegression()


log.fit(X_train , y_train)


log_pred =log.predict(X_test)



print("the accuracy resutl is the: \n", accuracy_score(y_test, log_pred))



from sklearn.tree import  DecisionTreeClassifier

from sklearn.metrics import accuracy_score


tree =DecisionTreeClassifier()


tree.fit(X_train , y_train)


tree_pred =tree.predict(X_test)


print("the accuracy result is the: \n", accuracy_score(y_test, tree_pred))



from sklearn.preprocessing  import StandardScaler


from sklearn.model_selection import train_test_split


stander =StandardScaler()


fuck = stander.fit_transform(X)



print(fuck)

X_train, X_test , y_train , y_test =train_test_split(X, y, test_size=0.2)


from  sklearn.tree  import   DecisionTreeClassifier

from sklearn.metrics import accuracy_score

fake_model =DecisionTreeClassifier()


fake_model.fit(X_train , y_train)


print(fake_model)



# the chick the over fitting the under fitting  : 
print('the chick the  train:\n', accuracy_score(y_train,  fake_model.predict(X_train)))


print('the  chick the   test \n',  accuracy_score(y_test ,  fake_model.predict(X_test)))




from sklearn.metrics  import precision_score, recall_score, f1_score




# Get predictions first
y_pred = fake_model.predict(X_test)

# Then calculate metrics
print('the precision score: \n', precision_score(y_test, y_pred))
print('the recall score: \n', recall_score(y_test, y_pred))
print('the f1 score: \n', f1_score(y_test, y_pred))












