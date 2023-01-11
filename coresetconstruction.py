from dataloader import Dataloader
from kmeanspp import kmeanspp
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

a = 1

def coreset_construction(points, centers, epsilon=0.05):
    S = []
    point_weights = dict()
    distances = euclidean_distances(points, centers)
    r = np.sqrt(np.sum(np.power(distances, 2))/(a*np.log(len(points))*len(points)))
    closest_centers = np.argmin(distances, 1)
    num_points = 0

    # I formulated point_weight s.t. its:
    #   key = cord of the center of the grid
    #   value = weight 
    for j in range(1):
        s = epsilon * 2**j * r / (np.sqrt(points.shape[1]))
        for i, (point, center_index) in enumerate(zip(points, closest_centers)):
            num_points += 1
            c = centers[center_index]
            grid_position = tuple(np.floor((point - c)/s))

            if grid_position in point_weights:
                point_weights[grid_position] += 1
            else: 
                point_weights[grid_position] = 1

    print(f"num_points: {num_points}")
    return point_weights


    print(closest_centers)
    print(r)


def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=10 ,show=False)
    centers, indices = kplusplus(coords, k, show=False)
    point_weights = coreset_construction(coords, centers)

if __name__ == '__main__':
    main()