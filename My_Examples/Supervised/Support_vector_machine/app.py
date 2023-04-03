
# https://scikit-learn.org/stable/modules/svm.html
# Where it use This 
# Image clasification 
#  object detection
#  protin clasification - BIO
# Degital charecter recognization

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

df =pd.DataFrame(iris.data,columns=iris.feature_names)


# --------------- JOVA MATE ---------------
# df0=df[:50]
# df1=df[50:100]
# df2=df[100:]

# import matplotlib.pylab as plt
# plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color="green",marker='+',label="A")
# plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color="red",marker='*',label="B")
# plt.xlabel("Sepal lenghth")
# plt.ylabel("Sepal width")
# plt.legend()
# plt.show()

from  sklearn.model_selection import train_test_split
x = df
# print(x)
df['target'] = iris.target
y = df['target']
# print(y)

x_tarin,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

# print(x_test)
from sklearn.svm import SVC

# Model = SVC(C=2) # if score is less then 1 then try to close near then 1 using changing C=1-50
# high value means hight  acuurcy

# Model = SVC(gamma=0) # try to change 0 value and watch score to high accurcy 

# Model = SVC(kernel='precomputed') #SHow image 
Model = SVC()
Model.fit(x_tarin,y_train)
print(Model.score(x_test, y_test))