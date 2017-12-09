from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # step 1: load data
    print "step 1: load data..."
    dataSet = []
    fileIn = open('color.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        dataSet.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])

    # run k means on different k
    for k in range(6, 7):
        clf = KMeans(n_clusters=k)  # set k
        s = clf.fit(dataSet)  # load dataset
        numSamples = len(dataSet)
        centroids = clf.labels_
        print centroids, type(centroids)  # show centroid
        print clf.inertia_  # show clustering effect
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr'] # color map
        x = []
        y = []
        z = []
        color_list = []
        for i in xrange(1000):
            rgb = []
            x.append(dataSet[i][0])
            y.append(dataSet[i][1])
            z.append(dataSet[i][2])
            rgb.append(x[i])
            rgb.append(y[i])
            rgb.append(z[i])
            color_list.append(rgb)
        # show dots with identical colors if they belong to the same class
        for i in xrange(numSamples):
            ax = plt.subplot(111, projection='3d')  # initialize a 3d plot project
            # # color dots by it rgb value
            # cdict = {}
            # cdict['red'] = x
            # cdict['green'] = y
            # cdict['blue'] = z
            C = np.array(color_list)
            ax.scatter(x[::], y[::], z[::], c=C/255.0) # plot data point with by rgb values
            ax.set_zlabel('Z')  # axis label
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # plot centroid with special shapes
        centroids = clf.cluster_centers_
        for i in range(k):
            ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2], c='red', s=70)
            # plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
            print '%f\t%f\t%f\n' % (centroids[i, 0], centroids[i, 1], centroids[i, 2]) # print centroid info
        plt.show()