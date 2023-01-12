from dataloader import Dataloader
from kmeanspp import kmeanspp
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


a = 1
n = 1000

def coreset_construction(points, centers, epsilon=1e-1):
    
    S = []
    point_weights = dict()
    d = points.shape[1]

    z = np.log(n) * np.log(a * np.log(n))
    distances = euclidean_distances(points, centers)
    r = np.sqrt(
            np.sum(np.power(distances, 2))/(a * np.log(n) * n)
            )

    closest_centers = np.argmin(distances, 1)
    min_distances = np.min(distances, axis=1)
    print(f"r: {r}")
    # I formulated point_weight s.t. its:
    #   key = cord of the center of the grid
    #   value = weight 

    print(f"ITER {0} --------------------------------")
    s = epsilon * r / (np.sqrt(d))
    distances = euclidean_distances(points, centers)
    closest_centers = np.argmin(distances, 1)
    min_distances = np.min(distances, axis=1)
    current_points = points[min_distances < r]
    current_centers = closest_centers[min_distances < r]
    
    print(f"points: {points.shape}")
    print(f"min_distances: {min_distances.shape}")
    print(f"current_centers.shape: {current_centers.shape}")

    for i, (point, center_index) in enumerate(zip(current_points, current_centers)):
        c = centers[center_index]
        grid_position = tuple(np.floor((point)/s))

        if grid_position in point_weights:
            point_weights[grid_position][1] += 1
        else: 
            point_weights[grid_position] = [point, 1]

    for j in range(1, int(z)+1):
        print(f"ITER {j} --------------------------------")
        s = epsilon * 2**j * r / (np.sqrt(d))
        distances = euclidean_distances(points, centers)
        closest_centers = np.argmin(distances, 1)
        min_distances = np.min(distances, axis=1)
        current_points = points[min_distances < 2**j*r and min_distances >= 2**(j-1)*r]
        current_centers = closest_centers[min_distances < 2**j*r and min_distances >= 2**(j-1)*r]
        
        print(f"points: {points.shape}")
        print(f"min_distances: {min_distances.shape}")
        print(f"current_centers.shape: {current_centers.shape}")

        for i, (point, center_index) in enumerate(zip(current_points, current_centers)):
            c = centers[center_index]
            grid_position = tuple(np.floor((point)/s))

            if grid_position in point_weights:
                point_weights[grid_position][1] += 1
            else: 
                point_weights[grid_position] = [point, 1]
                # print(f"point_weights[grid_position]: {point_weights[grid_position]}")
        x = np.array(list(point_weights.values()), dtype=object)
        print(f"lolsad : \n{np.array(x.shape)}")
        break
        # # print(f"lolsad : {np.array(list(point_weights.values()), dtype=object)[:][0][0]}")
        # sns.scatterplot(current_points[:, 0], current_points[:, 1], palette="deep")
        # plt.scatter(centers[:, 0], centers[:, 1], cmap="b")
        # plt.scatter(np.array(list(point_weights.values()))[:,0], np.array(list(point_weights.values()))[:,0,1], cmap="r")
        # plt.grid()
        # plt.show()
        # plt.close()
    
    return point_weights

def run_coreset_construction(points, k, epsilon=0.05):
    points = np.array(list(points)) # Needed for parallelization
    centers, indices = kmeanspp(points, k, show=False)
    point_weights = coreset_construction(points, centers)
    print("lol")

def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=n ,show=False)
    run_coreset_construction(coords, k)
    # point_weights = coreset_construction(coords, centers)

if __name__ == '__main__':
    main()