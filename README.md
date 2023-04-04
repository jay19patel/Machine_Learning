# Machine_Learning

### Scikit-learn is a popular machine learning library in Python that provides a wide range of machine learning models and algorithms. Here is a brief description of some of the most commonly used models in scikit-learn and their applications:

## -  Linear Regression:
 Used for regression tasks where the relationship between the input variables and the output variable is linear.
    - example: it can be used to quantify the relative impacts of age, gender, and diet (the predictor variables) on height (the outcome variable).

## - Logistic Regression: 
Used for classification tasks where the output variable is categorical.
    - example : Email Filter spam or  notspam , Fail or pass, O or 1 yes or not

## -  Support Vector Machines (SVM): Used for both classification and regression tasks, particularly when the data has complex boundaries.
    - one of the most popular Supervised Learning algorithms,Classification as well as Regression problems
    - lot of training data 

## - K-Nearest Neighbors (KNN): K-NN algorithm stores all the available data and classifies a new data point based on the similarity.
    - K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
    - stores all the available data and classifies a new data point based on the similarity

## - Naive Bayes: Used for classification tasks where the input variables are independent and the output variable is categorical.
    - It is mainly used in text,image classification that includes a high-dimensional training dataset.
    - example: spam filtration, Sentimental analysis, and classifying articles.

## - Decision Trees: Used for both classification and regression tasks where the data has a hierarchical structure.
    - A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.
    - 

## - Random Forest: Used for classification and regression tasks where the data has many dimensions and is prone to overfitting.
    - divdide in multiple group and give output
    - matching all grup anser in the end and then return anser 
    - Example : Banking: Banking sector mostly uses this algorithm for the identification of loan risk.

## - Neural Networks: Artificial Neural Network is biologically inspired by the neural network, which constitutes after the human brain..

## - Clustering Algorithms: Used for unsupervised learning tasks where the goal is to find patterns in the data without prior knowledge of the output variable.
    - A way of grouping the data points into different clusters, consisting of similar data points. The objects with the possible similarities remain in a group that has less or no similarities with another group. 
    - EXAMPLE :
            Market Segmentation
            Statistical data analysis
            social network analysis
            Image segmentation
        It is used by the Amazon in its recommendation system to provide the recommendations as per the past search of products. Netflix also uses this technique to recommend the movies and web-series to its users as per the watch history.

## LabelEncoder  For Covert Labeling into Numbers 

 - User for labeling a datas like:
 
```python

#    company           job           degree   

# 0  google      sales executive  bachelors                      
# 1  google      sales executive    masters                      
# 3  google     business manager    masters                     
# 4  google  computer programmer  bachelors 



from sklearn.preprocessing import LabelEncoder 

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

# Je pan name multiple time replate thay data ma to ene change kari ne 
#  number ma convert kari devanu jemke ("Jay":0,"Salman":1,"Jetha":2)
``` 


## Confusion_matrix 
- this is use for vesioultize how many time our model predict which number vs real number  
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)

# good visiultion 
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
```

## Save model in local storage 

- This is use where we dont need to again and again train our model
- if me have milions of data then so much time contain to train if we try all test to train then so much time conume thats why save the train model then we use train data to fast execution 
```python
import joblib
#  Import data in file 
joblib.dump(model,'file_path')

# Export data from File 

mj = joblib.load('file_path')
```

## Conver data in 0-1 Ranges 
```python 
from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()
mx.fit(df[['Income']])
df['Income']=mx.transform(df[['Income']])

mx.fit(df[['Age']])
df['Age']=mx.transform(df[['Age']])

# print(df.head())
```
## Conert each word into row NL word using NLTK
```python


# Netureal Langue function using NLTK 
# Natural Language Toolkit
#   normla to convert root word 
# import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def mynltk(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# print(ps.stem('salmankhan'))
my_df['mytags'] = my_df['mytags'].apply(mynltk)

```
