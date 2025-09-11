import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import plot_tree

def main():
    pd.set_option('display.max_column', None)

    X, Y = make_classification(n_features=5, n_redundant=0, n_informative=5, n_clusters_per_class=1)

    df = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5'])
    df['target'] = Y

    # BAGGING
    bag = BaggingClassifier(max_features=2)
    bag.fit(df.iloc[:,:5], df.iloc[:,-1])

    plt.figure(figsize=(10,10))
    plot_tree(bag.estimators_[0])
    plt.show()

    # RANDOM FOREST
    rf = RandomForestClassifier(max_features=2)
    rf.fit(df.iloc[:,:5], df.iloc[:,-1])

    plt.figure(figsize=(10,10))
    plot_tree(rf.estimators_[0])
    plt.show()

    
if __name__ == "__main__":
    main()