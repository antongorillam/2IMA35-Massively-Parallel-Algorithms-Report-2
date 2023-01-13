from dataloader import Dataloader
from kmeanspp import kmeanspp
from sklearn.metrics.pairwise import euclidean_distances
from pyspark.sql.session import SparkSession
spark = SparkSession.builder.appName("DFTest").getOrCreate()
sc = spark.sparkContext
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

a = 1
n = 1000000
k = 3
EPSILON = 1e-1
NUM_MACHINES = [16, 8, 4, 2]

def coreset_construction(points, weights, centers, epsilon=1e-1):
    
    point_weights = dict()
    d = points.shape[1]

    z = np.log(n) * np.log(a * np.log(n))
    distances = euclidean_distances(points, centers)
    r = np.sqrt(
            np.sum(np.power(distances, 2))/(a * np.log(n) * n)
            )

    closest_centers = np.argmin(distances, 1)
    min_distances = np.min(distances, axis=1)
    s = epsilon * r / (np.sqrt(d))
    distances = euclidean_distances(points, centers)
    closest_centers = np.argmin(distances, 1)
    min_distances = np.min(distances, axis=1)
    current_points = points[min_distances < r]
    current_weights = weights[min_distances < r]
    current_centers = closest_centers[min_distances < r]

    
    cost = np.sum(np.power(min_distances, 2))

    for i, (point, weight, center_index) in enumerate(zip(current_points, current_weights, current_centers)):
        c = centers[center_index]
        grid_position = tuple(np.floor((point)/s))

        if grid_position in point_weights:
            point_weights[grid_position][1] += weight
        else: 
            point_weights[grid_position] = [point, weight]

    for j in range(1, int(z)+1):
        s = epsilon * 2**j * r / (np.sqrt(d))
        distances = euclidean_distances(points, centers)
        closest_centers = np.argmin(distances, 1)
        min_distances = np.min(distances, axis=1)
        current_points = points[np.logical_and(min_distances >= 2 ** (j - 1) * r, min_distances < 2 ** j * r)]
        current_weights = weights[np.logical_and(min_distances >= 2 ** (j - 1) * r, min_distances < 2 ** j * r)]
        current_centers = closest_centers[np.logical_and(min_distances >= 2 ** (j - 1) * r, min_distances < 2 ** j * r)]

        if current_points.shape[0] == 0:
            break
        
        # print(f"current points: {current_points.shape}")
        # print(f"min_distances: {min_distances.shape}")
        # print(f"current_centers.shape: {current_centers.shape}")

        for i, (point, weight, center_index) in enumerate(zip(current_points, current_weights, current_centers)):
            c = centers[center_index]
            grid_position = tuple(np.floor((point)/s))

            if grid_position in point_weights:
                point_weights[grid_position][1] += weight
            else: 
                point_weights[grid_position] = [point, weight]
        
    temporary_set = np.array(list(point_weights.values()), dtype=object)
    S_weights = np.array([i for i in temporary_set[:,1]])
    S = np.array([i for i in temporary_set[:,0]])
    coreset_dist = euclidean_distances(S, centers).min(axis=1)
    coreset_cost = np.sum(np.power(coreset_dist, 2) * S_weights)

    # print(sum(S_weights))
    # print(f"temporary_set: {temporary_set.shape}")
    # print(f"point_weights: {point_weights}")
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

def run_coreset_construction(i, points_weights, k=k, epsilon=EPSILON):
    points_weights = np.array(list(points_weights))[0] # Needed for parallelization
    points, weights = points_weights[:,:2], points_weights[:,2:3]
    centers, indices = kmeanspp(points, k, show=False)
    new_point, new_weight = coreset_construction(points, weights, centers, epsilon=epsilon)
    new_point_weight = np.concatenate([new_point, new_weight], axis=1)
    yield i, new_point_weight

def mapper_assign_index(element, num_machines):
    index = np.random.randint(num_machines)
    return (index, element)

def coreset_construction_parallel(coords):
    weights = np.array([1 for _, _ in enumerate(coords)]).reshape(-1, 1)
    point_weights = np.concatenate([coords, weights], axis=1)
    # Assign indices to machines
    index_rdd = sc.parallelize(point_weights, NUM_MACHINES[0])\
        .zipWithIndex()\
        .map(lambda x : [x[0], x[1] % NUM_MACHINES[0]])
    mapped_points = np.array(index_rdd.collect())

    coreset = []
    # Split it so each indivdual machine is in a list
    for machine_index in range(NUM_MACHINES[0]):
        filtered_points = mapped_points[mapped_points[:,1] == machine_index]
        filtered_points = np.array([list(i[0]) for i in filtered_points])
        coreset.append(filtered_points)

    for num_machines in NUM_MACHINES:
        rdd = sc.parallelize(coreset, num_machines)
        points = rdd.mapPartitionsWithIndex(run_coreset_construction, preservesPartitioning=True).collect()

        # Merge every adjecent
        coreset = np.array([np.concatenate([points[i][1], points[i+1][1]], axis=0) for i in range(0, len(points), 2)])
    return coreset[0]

def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=n, show=False)
    d = coords.shape[1]
    coreset = coreset_construction_parallel(coords)

    print(f"sum of all weights {coreset[:,2].sum()} and the total sum of all points {coords.shape[0]}")
    print(f"coords size: {coords.shape[0]}, coreset size: {coreset.shape[0]}")
    plt.scatter(coords[:,0], coords[:,1], cmap="r", label="Original")
    plt.scatter(coreset[:, 0], coreset[:, 1], cmap="r", label="Coreset")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    

if __name__ == '__main__':
    main()