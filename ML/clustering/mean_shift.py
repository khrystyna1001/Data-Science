import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./mall.csv')

    X = df.iloc[:,[3,4]].values

    # FITTING MEAN SHIFT CLUSTERING
    ms = MeanShift(bandwidth=20)

    # ADJUST BANDWIDTH TO INFLUENCE CLUSTERS
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    #VISUALIZING
    n_clusters = len(np.unique(labels))
    colors = ["red", "blue", "green", "yellow", "cyan", "pink", "purple"]

    for i in range(n_clusters):
        cluster = X[labels == i]
        plt.scatter(cluster[:,0], cluster[:,1], s=100, c=colors[i], label=f"C{i+1}")
    plt.title('Cluster of Clients')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score (0-100)')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()