from statistics import mean

# find leaniger line where slop is slop of line and b is where line meet at Y axis(intercept)
def Liner_line(xs,ys):
    slop =(((mean(xs)*mean(ys)) - mean(xs-ys) )/ ((mean(xs) - mean(xs)) - mean(xs*xs)) )
    b = mean(ys) - slop*(mean(xs))
    return slop,b



# means find 
import pandas as pd

# read csv 
df = pd.read_csv('Tests/regration_line/data.csv')
# print(df)

# watch mean of datas 
print(df.describe())

#get only male datas 
male_data=df[df['GENDER']=='M']

#get only female datas 
female_data =df[df['GENDER']=='F']

# print("------male data ----------")
# print(male_data)
# print("------Female data ----------")
# print(female_data)


# only get Heights from csv file as a list Formate
Height_Data = male_data['HEIGHT'].tolist()
Weight_Data = male_data['WEIGHT'].tolist()
 
import numpy as np

xax=np.array(Height_Data,dtype=np.float64)
yax=np.array(Weight_Data,dtype=np.float64)
my_slop,my_b = Liner_line(xax,yax)
reg_line =list(((my_slop*x) + my_b for x in xax))
xavrage = mean(xax)
yavrage = mean(yax)
# here  errror :("if  error have some id type that means our data give multiple data so try loop or list")
# for i in reg_line:
#     print(i)
import matplotlib.pyplot as plt

plt.scatter(xax,yax)
plt.scatter(xavrage,yavrage,label="Avrage",color="black",s=100) #allover Avrage Prediction
plt.plot(xax,reg_line,label="Regartion Line")
plt.title("Height and Weight Graph")
plt.ylabel("Weight")
plt.xlabel("Height")
plt.legend()
plt.show()