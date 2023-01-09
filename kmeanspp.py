from dataloader import Dataloader
from sklearn.cluster import kmeans_plusplus
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Implement the coreset construction algorithm


def kplusplus(coords, k, show=False):
    if isinstance(k, int):
        centers, indices = kmeans_plusplus(coords, k)
    else:
        centers, indices = kmeans_plusplus(coords, len(set(k)))
    if show:
        sns.scatterplot(centers[:, 0], centers[:, 1], [3] * len(centers), palette="deep")
        plt.grid()
        plt.show()
        plt.close()
    return centers, indices

def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=10000 ,show=False)
    centers, indices = kplusplus(coords, k, True)
    sns.scatterplot(coords[:, 0], coords[:, 1], k)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()