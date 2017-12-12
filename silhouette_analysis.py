import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# to determine which k is optimal for K-Means

print "load data..."
dataSet = []
fileIn = open('color.csv')
for line in fileIn.readlines():
    lineArr = line.strip().split(',')
    dataSet.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])

range_n_clusters = [3]

for n_clusters in range_n_clusters:
    clf = KMeans(n_clusters=n_clusters)
    labels = clf.fit_predict(dataSet)
    silhouette_avg = silhouette_score(dataSet, labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
