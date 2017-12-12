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

data_array = np.array(dataSet)
print type(data_array[0][1])

# dataSet.append([1,2,3])
# dataSet.append([1,3,4])
# dataSet.append([5,5,6])
range_n_clusters = [3]

# print data_array

for n_clusters in range_n_clusters:
    clf = KMeans(n_clusters=n_clusters)
    labels = clf.fit_predict(data_array)
    silhouette_avg = silhouette_score(data_array, labels)
    print 'For n_clusters = %d, The average silhouette_score is : %f' % (n_clusters, silhouette_avg)
