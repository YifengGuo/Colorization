from sklearn.cluster import KMeans
import numpy as np


if __name__ == '__main__':
    # step 1: load data
    print "load data..."
    dataSet = []
    fileIn = open('color.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        dataSet.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])

    clf = KMeans(n_clusters=6)
    s = clf.fit_predict(dataSet)
    labels = clf.labels_
    centroids = clf.cluster_centers_
    print centroids

    temp = dataSet

    for i in range(len(temp)):
        temp[i] = centroids[labels[i]]
        with open('color_labels.txt', 'a') as f:
            f.write('\n' + str(temp[i]))
       # print temp_40000[i]

    # for data in temp_40000:
    #     print data
