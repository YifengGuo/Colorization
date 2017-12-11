from sklearn.cluster import KMeans
import numpy as np

if __name__ == '__main__':
    print 'load data'
    dataSet = []
    fileIn = open('input.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        dataSet.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]),
                        float(lineArr[3]), float(lineArr[4]), float(lineArr[5]),
                        float(lineArr[6]), float(lineArr[7]), float(lineArr[8])])


    clf = KMeans()
    s = clf.fit_predict(dataSet)
    centroids = clf.cluster_centers_
    print centroids

    # for i in range(len(centroids)):
    #     print centroids[i][4]