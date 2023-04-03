import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score

df= pd.read_csv('My_Examples\(LR)home_predict_price\data.csv')
# print(df.head())

newdf = df.drop('Price',axis='columns') #features 
model = linear_model.LinearRegression()  
model.fit(newdf,df.Price)
print(model.predict([[1,5]]))



# get_df= pd.read_csv('test.csv')

# # here add your own data using 2d array : [[10]]
# p = model.predict(get_df)

# get_df['Price']=p
# print(get_df)

# # get_df.to_csv("New.csv")
