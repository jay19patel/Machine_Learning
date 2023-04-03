
# ------------------------------- K FOLD ------------------------------------

# alag  alag technic user kari ne score jovano
#  je haru perform kare te try karvanu 
# pan ama problem e ke jo test data change thay to scre vadh ghat thay 
# 

# K fold hu kare ke 10-15 var random data let test mate ne pasi tene score avrage kadi ne pasi Model kayu haru te key

# from sklearn.model_selection import KFold
# kf = KFold(n_splits=3)
# print(kf)
# for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
#     print(train_index,test_index)
# Out Put :n_splits=3  3bhag  kari dey ne badhani value alag alag 
# [3 4 5 6 7 8] [0 1 2]
# [0 1 2 6 7 8] [3 4 5]
# [0 1 2 3 4 5] [6 7 8]

# Out Put :n_splits=2
# [5 6 7 8] [0 1 2 3 4]
# [0 1 2 3 4] [5 6 7 8]
# -------------------------------------------------------------------

# ------------------ MULTI MODEL SCORE FINDER -----------------------------

# def get_score(model,x_train,x_test,y_train,y_test):
#     model.fit(x_train,y_train)
#     score = model.score(x_test,y_test)
#     return score


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_digits
# import pandas as pd 
# degits = load_digits()
# x_train,x_test,y_train,y_test = train_test_split(degits.data,degits.target,test_size=0.3)
# ans = get_score(SVC(),x_train,x_test,y_train,y_test)
# print(ans)
# -------------------------------------------------------------------


# ---------------------- K FOLD FIND SCORE -----------------------
# from sklearn.model_selection import StratifiedKFold
# from sklearn.datasets import load_digits
# degits = load_digits()
# skf = StratifiedKFold(n_splits=3)
# score_Logistic =[]
# score_svm =[]
# score_rf =[]


# def get_score(model,x_train,x_test,y_train,y_test):
#     model.fit(x_train,y_train)
#     score = model.score(x_test,y_test)
#     return score

# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# for train_index,test_index in skf.split(degits.data,degits.target):
#     x_train,x_test,y_train,y_test = degits.data[train_index],degits.data[test_index],degits.target[train_index],degits.target[test_index]
#     ans1 = get_score(LogisticRegression(),x_train,x_test,y_train,y_test)
#     ans2 = get_score(SVC(),x_train,x_test,y_train,y_test)
#     ans3 = get_score(RandomForestClassifier(),x_train,x_test,y_train,y_test)
#     score_Logistic.append(ans1)
#     score_svm.append(ans2)
#     score_rf.append(ans3)
# from statistics import mean
# print("Logstic Regration : ",mean(score_Logistic))
# print("SVM : ",mean(score_svm))
# print("Random Forest: ",mean(score_rf))
# -------------------------------------------------------------------


# -------- Sort cut Methods (Cross Val Score ) -------------------
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
degits = load_digits()
model = cross_val_score(RandomForestClassifier(n_estimators=200),degits.data,degits.target,cv=5)
# n_estimators  = Tree NUmber (ketla tree ma divide thava joyye e )
#cv = Fold NUmber (atle ke ketla fold joye te pramane set karvu)
from statistics import mean
print(mean(model))

# -------------------------------------------------------------------
