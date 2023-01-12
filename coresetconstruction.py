from dataloader import Dataloader
from kmeanspp import kmeanspp
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


a = 1

def coreset_construction(points, centers, epsilon=1e-1):
    
    S = []
    point_weights = dict()
    n = points.shape[0]
    d = points.shape[1]

    z = np.log(n) * np.log(a * np.log(n))
    distances = euclidean_distances(points, centers)
    r = np.sqrt(
            np.sum(np.power(distances, 2))/(a * np.log(n) * n)
            )
    closest_centers = np.argmin(distances, 1)

    # I formulated point_weight s.t. its:
    #   key = cord of the center of the grid
    #   value = weight 
    for j in range(1):
        s = epsilon * 2**j * r / (np.sqrt(d))
        new_distances = distances[distances.min(axis=1) < 2]
        for i, (point, center_index) in enumerate(zip(points, closest_centers)):
            
            
            c = centers[center_index]
            grid_position = tuple(np.floor((point-c)/s))

            if grid_position in point_weights:
                point_weights[grid_position] += 1
            else: 
                point_weights[grid_position] = 1

        sns.scatterplot(points[:, 0], points[:, 1], palette="deep")
        plt.scatter(centers[:, 0], centers[:, 1], cmap="b")
        plt.scatter(np.array(list(point_weights.keys()))[:,0], np.array(list(point_weights.keys()))[:,1], cmap="r")
        plt.grid()
        plt.show()
        plt.close()
        return point_weights

def run_coreset_construction(points, k, epsilon=0.05):
    points = np.array(list(points)) # Needed for parallelization
    centers, indices = kmeanspp(points, k, show=False)
    point_weights = coreset_construction(points, centers)
    print("lol")

def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=1000 ,show=False)
    run_coreset_construction(coords, k)
    # point_weights = coreset_construction(coords, centers)

if __name__ == '__main__':
    main()