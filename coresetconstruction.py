from dataloader import Dataloader
from kmeanspp import kplusplus
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def coresetConstruction(points, centers, epsilon=0.05):
    S = []
    distances = euclidean_distances(points, centers)

    r = np.sqrt(np.sum(np.power(distances, 2))/(np.log(len(points)*len(points))))

    closestCenters = np.argmin(distances, 1)
    print(closestCenters)
    print(r)


def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=10 ,show=False)
    centers, indices = kplusplus(coords, k, show=False)
    coresetConstruction(coords, centers)

if __name__ == '__main__':
    main()