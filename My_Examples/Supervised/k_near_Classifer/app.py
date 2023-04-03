from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# simple premodel ml model 

iris = datasets.load_iris()

x = iris['data']
y = iris['target']

print(x[0],y[0])

clf = KNeighborsClassifier()
clf.fit(x,y)

predict = clf.predict([[5,1,1,10]])

if predict == 0:
    print("Jay")
else :
    print(" Vijay")



## KNearest_neaighber
# je pan new value appa te ne check karse je pan ena najik ma hase te jose
# chek kari ne ganse ke 1 chhe ke 2 chhe ke 3 and te pramane gane and je vadhare rey najik te predict kari ly