# <!-- Use  Decidion Tree -->
#  qution puse and te pramane agad jay 
#  gharebthi amdavad javanu rey to pella bux ke rixa ma gharethi bus stand kato rail station te nakki thay
#  bus to pasi bus ni tikit
#  train to train ni tiki
#  be  mathi choose kari ne pasi agad or 
#  aam decidion tree step by step afad jay
# 10-15 jan ne puvanu pasi je ghana aser ape te pramane javanu 

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()

df = pd.DataFrame(digits.data)
# print(df)

import matplotlib.pyplot as plt
# plt.matshow(data.images[1])
# plt.show()

df['target'] = digits.target # add columns in data table 

X = df.drop('target',axis='columns')
Y = df.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100) # change this value and incress scoree
model.fit(x_train,y_train)

# print(model.score(x_test,y_test))

y_predict = model.predict(x_test)
# print(y_predict)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
# print(cm)

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()