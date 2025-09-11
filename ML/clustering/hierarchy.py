import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./mall.csv')

    X = df.iloc[:,[3,4]].values

    # DENDROGRAM
    dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))

    plt.title('Dendrograms')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean Distance')
    plt.show()

    # CLUSTERING
    hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage="ward")
    y_hc = hc.fit_predict(X)

    plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=100, c="red", label="Cluster1")
    plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s=100, c="blue", label="Cluster2")
    plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s=100, c="green", label="Cluster3")
    plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s=100, c="yellow", label="Cluster4")
    # plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s=100, c="cyan", label="Cluster5")
    # plt.scatter(X[y_hc == 5,0], X[y_hc == 5,1], s=100, c="purple", label="Cluster6")
    # plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='black', label='Centroids')
    plt.title('Cluster of Clients')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score (0-100)')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()