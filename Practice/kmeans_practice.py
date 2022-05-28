import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

# EXERCISE :
# Use iris flower dataset from sklearn library and try to form clusters of flowers using
# petal width and length features. Drop other two features for simplicity.
# Figure out if any preprocessing such as scaling would help here
# Draw elbow plot and from that figure out optimal value of k


iris = load_iris()
data =pd.DataFrame(iris.data, columns=iris.feature_names)
data = data.drop(["sepal length (cm)", "sepal width (cm)"], axis =1)

# plt.scatter(data["petal length (cm)"], data["petal width (cm)"])
# plt.show()


# Scaling
def scale_data(data, feature1, feature2):
    scaler = MinMaxScaler()
    temp_data =  scaler.fit_transform(data)
    data[feature1], data[feature2] = temp_data[:, 0], temp_data[:, 1]
    return data

data = scale_data(data, "petal length (cm)", "petal width (cm)")

# Clustering
# 1. Finding valid val of K via Elbow method

sse_vals = []
kval_range = range(1, 20)

for k_values in kval_range:
    kmeans = KMeans(n_clusters=k_values)
    kmeans.fit(data.values)
    sse_vals.append(kmeans.inertia_)


# So after plotting the elbow-techq graph, we get the proper value of K-means to be : 3
# plt.plot(kval_range, sse_vals)
# plt.show()


kmeans_final = KMeans(n_clusters=3)
data["clusters"] = kmeans_final.fit_predict(data.values)
clr_pallete = ["#D90368", "#FB8B24", "#04A777"]
sns.scatterplot(data["petal length (cm)"], data["petal width (cm)"], hue=data["clusters"], markers=".", palette=clr_pallete)
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], marker='X', color = 'black')
plt.show()
