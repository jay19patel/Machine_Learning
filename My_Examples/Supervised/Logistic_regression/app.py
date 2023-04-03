from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.head())

data =df.drop('smoker',axis='columns')
x = data[['age']]
y=df['smoker']
# plt.scatter(df['age'],df['smoker']) 
# plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
Model = LogisticRegression()
Model.fit(x_train,y_train)

# print(x_test)
Result = Model.predict([[22]])
proba = Model.predict_proba([[22]])
print(proba)
print(Result)


