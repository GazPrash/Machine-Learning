# K is a free parameter 
# (video ref : https://www.youtube.com/watch?v=EItlUEPCIzM&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=14)

# For unsupervised learning, with a dataset only containing features and no targets
# You start with a random value for K, let's say 2, then you put K random points
# on the graph representing K different centroids for K different clusters in the
# datasets.

# Now these centroids at random points won't work, so we calculate their distance from different
# points and adjust them accordingly. Kind of like making the centroid the center of mass 
# of these discrete points 
# 
#                  ---->centroid at random starting place. (put it at COM)
#                  | 
#             | .  * .
#             | . .    .
# -------------------------------
#             |
#             |
# 
# We re-compute and keep adjusting the clusters untill we reach a point where re-adjusting doesn't affect the clusters
# hence final clusters will be formed at the end.


# How to come up with a value off K? (Possible number of Clusters in your datasets)
# 1). Elbow Method : Calc. Sum of Squared Errors
# SSE(Total) = SSE1(Cluster1) + SSE2(Cluster2) + ..... + SSEn(Cluster(n))
# Then we plot SSEt against the values of K (in the above case from 1 to n) and we will achieve a strictly decrs. graph
# Basically in this plot, we aim for the elbow-point in the plot (Refer to Elbow_Technique.png)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv"
)

# plt.scatter(data["Age"], data["Income($)"])
# plt.show()

kmns = KMeans(n_clusters=3)
pred = kmns.fit_predict(data[["Age", "Income($)"]])
data["clusters"] = pred

# palette = ['tab:blue', 'tab:orange', 'tab:red']
# sns.scatterplot(x = data["Age"], y = "Income($)", hue="clusters", data = data, palette=palette)
# plt.show()

# This produces a scatter plot but we can see the clusters are not properly distributed,
# this is because it's not scaled properly as y is incredibly large in range as compared to the values
# on the x-axis. So we use MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["Age", "Income($)"]])
data["scaled_age"], data["scaled_income"] = scaled_data[:, 0], scaled_data[:, 1]

scaled_kms = KMeans(n_clusters= 3)
data["scaled_clusters"] = scaled_kms.fit_predict(data[["scaled_age", "scaled_income"]])
centroids = scaled_kms.cluster_centers_

palette = ['tab:blue', 'tab:orange', 'tab:red']
sns.scatterplot(x = "scaled_age", y = "scaled_income", hue="scaled_clusters", data = data, palette=palette)
plt.scatter(centroids[:, 0], centroids[:, 1], color = 'black', marker='*')
plt.show()


# Finding starter K value using Elbow Plot method
SSE = []
for k_values in range(1, 13):
    kmeans = KMeans(n_clusters=k_values)
    kmeans.fit(data[["scaled_age", "scaled_income"]].values)
    SSE.append(kmeans.inertia_)

print(SSE)


# plt.plot(np.arange(1, 13), SSE)
# plt.show()






