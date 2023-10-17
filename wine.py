from sklearn.datasets import load_wine
data=load_wine()
x = data.data
y = data.target 

print(x.shape)
print(y.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
print(knn)
knn.fit(x, y)
knn.predict([[12.69,1.53,2.26,20.7,80,1.38,1.46,.58,1.62,3.05,.96,2.06,495]])
