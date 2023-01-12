from cProfile import label
from dataloader import Dataloader
from kmeanspp import kmeanspp
from sklearn.metrics.pairwise import euclidean_distances
from pyspark.sql.session import SparkSession
spark = SparkSession.builder.appName("DFTest").getOrCreate()
sc = spark.sparkContext
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

a = 1
n = 100000
NUM_MACHINES = 6

def coreset_construction(points, centers, epsilon=1e-1):

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
    
    cost = np.sum(np.power(min_distances, 2))

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
        current_points = points[np.logical_and(min_distances >= 2 ** (j - 1) * r, min_distances < 2 ** j * r)]
        current_centers = closest_centers[np.logical_and(min_distances >= 2 ** (j - 1) * r, min_distances < 2 ** j * r)]

        if current_points.shape[0] == 0:
            break
        
        print(f"current points: {current_points.shape}")
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
    temporary_set = np.array(list(point_weights.values()), dtype=object)
    S_weights = np.array([i for i in temporary_set[:,1]])
    S = np.array([i for i in temporary_set[:,0]])
    coreset_dist = euclidean_distances(S, centers).min(axis=1)
    coreset_cost = np.sum(np.power(coreset_dist, 2) * S_weights)

    # print(sum(S_weights))
    # print(S.shape)
    # print(centers.shape)
    # print(coreset_dist.shape)
    # print(f"cost: {cost}")
    # print(f"coreset_cost: {coreset_cost}")
    # print(f"bound: {(1-epsilon) * cost} less than {coreset_cost} less than {(1+epsilon) * cost}")

    # print(f"lolsad : {np.array(list(point_weights.values()), dtype=object)[:][0][0]}")
    # plt.scatter(points[:, 0], points[:, 1], cmap="g", label="Original")
    # plt.scatter(centers[:, 0], centers[:, 1], cmap="b", label="Centers")
    # plt.scatter(S[:,0], S[:,1], cmap="r", label="Coreset")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.close()
    
    return S, S_weights

def run_coreset_construction(points, k, epsilon=0.05):
    points = np.array(list(points)) # Needed for parallelization
    centers, indices = kmeanspp(points, k, show=False)
    point_weights = coreset_construction(points, centers)
    yield point_weights

def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=n, show=False)
    d = coords.shape[1]
    rdd = sc.parallelize(coords, NUM_MACHINES) 
    centers = rdd.mapPartitions(lambda x : run_coreset_construction(x, 3))
    point_weights = centers.collect()
    s, s_weight = np.zeros((0, d)), np.zeros((0, 1))
    for s_i, s_weight_i in range(point_weights):
        # print(f"s: {s}, s_weight: {s_weight} ")
        s = np.concatenate([s, s_i])
        s_weight = np.concatenate([s_weight, s_weight_i])

    print("öool")
if __name__ == '__main__':
    main()