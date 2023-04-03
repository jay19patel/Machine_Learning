# N number of feature 
#  Best Algo 
# Image classification use
#  Cluster dekhay jay to te ma most of case K NN user karvu 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
x = df.drop(['target'], axis='columns')
y = df.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


from sklearn.neighbors import KNeighborsClassifier
model  = KNeighborsClassifier(n_neighbors=9) 
# neighbors means ke 9 circule banave ne pasi chek kare ketla A ma chhe ne ketla B ma
# Je vadhare rey  A ke B te ptamane ene classdification karvama aave 
model.fit(x_train,y_train)
print(model.score(x_test,y_test))