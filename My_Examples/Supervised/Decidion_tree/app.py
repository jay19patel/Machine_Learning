import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn import tree
# name to numbber clasification("Jay":0,"Salman":1,"Jetha":2)
df = pd.read_csv('Decidion_tree/data.csv')
# print(df.head())

data_x = df.drop('salary_more_then_100k',axis='columns')
data_y = df['salary_more_then_100k']

# print(data_y.head())

x_company = LabelEncoder()
x_job = LabelEncoder() 
x_degree = LabelEncoder()


data_x['x_company']=x_company.fit_transform(data_x['company'])
data_x['x_job']=x_company.fit_transform(data_x['job'])
data_x['x_degree']=x_company.fit_transform(data_x['degree'])

X = data_x.drop(['company','job','degree'],axis='columns')
print(X.head())
#    x_company  x_job  x_degree
# 0          2      2         0
# 1          2      2         1
# 2          2      0         0
# 3          2      0         1
# 4          2      1         0


Model = tree.DecisionTreeClassifier()
Model.fit(X,data_y)

output = Model.predict([[0,2,0]])
print(output)



# line bye bye tree isabe scan kare ne out put aape 


#  qution puse and te pramane agad jay 
#  gharebthi amdavad javanu rey to pella bux ke rixa ma gharethi bus stand kato rail station te nakki thay
#  bus to pasi bus ni tikit
#  train to train ni tiki
#  be  mathi choose kari ne pasi agad or 
#  aam decidion tree step by step afad jay




#  badha ne pusvanu je ghana loko key te karvanu 