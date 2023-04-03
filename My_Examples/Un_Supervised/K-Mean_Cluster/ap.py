#  K-Means Cluster
#  data ne analysis kari ne pachi ene cluster ma group kari devanu 
import pandas as pd 
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
df = pd.read_csv('My_Examples/Un_Supervised/K-Mean_Cluster/data.csv')


## data ne convert karvanu 0-1 ni bache jethi graph sara dekhay 
from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()
mx.fit(df[['Income']])
df['Income']=mx.transform(df[['Income']])

mx.fit(df[['Age']])
df['Age']=mx.transform(df[['Age']])

# print(df.head())



# # divide person in cluster
km = KMeans(n_clusters=3)# n_clusters=3 couse  ploting ma 3 cluster chhe 
y_predict = km.fit_predict(df[['Age','Income']])
df['Cluster']= y_predict
# print(km.cluster_centers_) # badha cluster nu center x y sodhe 


print(df)
df1 = df[df.Cluster ==0]
df2 = df[df.Cluster ==1]
df3 = df[df.Cluster ==2]

plt.scatter(df1.Age,df1.Income,color ='green')
plt.scatter(df2.Age,df2.Income,color ='red')
plt.scatter(df3.Age,df3.Income,color ='blue')
plt.title("Clusters")
plt.show()


# SSE (Sum Squed error)
sse =[]
k_range = range(1,10)
for k in k_range:
    kn = KMeans(n_clusters=k)
    kn.fit(df[['Age','Income']])
    sse.append(kn.inertia_)

plt.plot(k_range,sse)
plt.title("SSE")
plt.show()



#  steps 
# data frame 
#  convert into 0 to 1 (MinMaxScaler) 
#  ploting karo
# badha cluster ne alag alag color apo 
# SSE find karo (hath ni koni na jem )