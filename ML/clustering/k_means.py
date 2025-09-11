import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./mall.csv')

    X = df.iloc[:,[3,4]].values

    # FOR EACH CLUSTER VALUE APPEND WCSS VALUE
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,11),wcss)
    plt.title('Elbow Method')
    plt.xlabel('Clusters')
    plt.ylabel('WCSS')
    plt.show()

    # APPLY K MEANS ALGORITHM TO DATASET
    kmeans = KMeans(n_clusters=6)
    y_kmeans = kmeans.fit_predict(X)

    plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=100, c="red", label="Cluster1")
    plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=100, c="blue", label="Cluster2")
    plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=100, c="green", label="Cluster3")
    plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s=100, c="yellow", label="Cluster4")
    plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s=100, c="cyan", label="Cluster5")
    plt.scatter(X[y_kmeans == 5,0], X[y_kmeans == 5,1], s=100, c="purple", label="Cluster6")
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='black', label='Centroids')
    plt.title('Cluster of Clients')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score (0-100)')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()