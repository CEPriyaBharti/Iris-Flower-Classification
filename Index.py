import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
from sklearn import preprocessing



df=pd.read_csv("irisf.csv")
df.head()


Species = {'Iris-setosa': 1,'Iris-versicolor': 2,'Iris-virginica': 3}
df.Species = [Species[item] for item in df.Species]
df.head()


df.hist(column='Species',bins=50)
df.columns

for col in df.columns: 
    print(col)


#x=df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']].Values
#x[0:5]

x=df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
x[0:5]


y=df['Species'].values
y[0:5]

x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x[0:5]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

print('Train Set :  ',x_train.shape,y_train.shape)
print('Test Set :  ',x_test.shape,y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k=3

neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
neigh

yhat=neigh.predict(x_test)
yhat[0:5]

from sklearn import metrics


print("Train Set Accuracy  :  ",metrics.accuracy_score(y_train,neigh.predict(x_train)))
print("Test set accuracy : ",metrics.accuracy_score(y_test,yhat))
