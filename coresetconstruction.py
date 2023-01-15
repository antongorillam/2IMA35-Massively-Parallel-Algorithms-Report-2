from dataloader import Dataloader
from kmeanspp import kmeanspp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from pyspark.sql.session import SparkSession
spark = SparkSession.builder.appName("DFTest").getOrCreate()
sc = spark.sparkContext
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

a = 1
n = 50000
N_SAMPLES_LIST = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
EPSILON_LIST = [1e-3, 1e-2, 0.05, 1e-1, 0.5]
K_LIST = [3, 4, 5, 6, 7, 8]
k = 3
NUM_MACHINES = [16, 8, 4, 2]
# EPSILON = 0.1

PLOT_FOLDER = "plots"


def coreset_construction(points, weights, centers, epsilon=0.1):
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
    # coreset_dist = euclidean_distances(S, centers).min(axis=1)
    # coreset_cost = np.sum(np.power(coreset_dist, 2) * S_weights)

    return S, S_weights

def blob_clustering(points_weights, k=k, epsilon=0.1, mode="non-parallel", show=False, path=None):
    
    start_time = time.time()
    points, weights = points_weights[:,:2], points_weights[:,2:3]
    initial_centers, indices = kmeanspp(points, k, show=False)
    if mode=="non-parallel":
        coreset_points, coreset_weights = coreset_construction(points, weights, initial_centers)
    elif mode=="parallel":
        coreset_points, coreset_weights = coreset_construction_parallel(points)
    
    # print(f"coreset_points: {coreset_points.shape}")
    # print(f"k: {k}")
    # kmeans_normal = KMeans(n_clusters=k, random_state=42).fit(points)
    kmeans_coreset = KMeans(n_clusters=k, random_state=42).fit(coreset_points)
    final_centers = kmeans_coreset.cluster_centers_

    center_labels = np.argmin(euclidean_distances(points, final_centers), 1)

    # normal_centers = kmeans_normal.cluster_centers_

    # final_centers = kmeans_coreset.cluster_centers_
    # coreset_center_labels = kmeans_coreset.labels_

    # coreset_dist = euclidean_distances(coreset_points, centers).min(axis=1)
    # coreset_cost = np.sum(np.power(coreset_dist, 2) * coreset_weights)

    # new_coreset_distances = euclidean_distances(coreset_points, final_centers)
    # new_closest_centers = np.argmin(new_coreset_distances, 1)

    if show:
        point_data = {
            "x": points[:, 0],
            "y": points[:, 1],
            "label": center_labels
        }

        center_data = {
            "x": final_centers[:, 0],
            "y": final_centers[:, 1]
        }

        df_normal = pd.DataFrame(point_data)
        df_centers = pd.DataFrame(center_data)

        plt.figure()
        # plt.grid()
        palette1 = sns.dark_palette("seagreen", k)
        # palette2 = sns.color_palette("deep", k)
        sns.scatterplot(data=df_normal, x="x", y="y", hue="label", palette=palette1, legend=False)
        # sns.scatterplot(data=df_coreset, x="x", y="y", hue="label", palette=palette1)
        # sns.scatterplot(data=df_centers, x="x", y="y", palette="red")
        # sns.scatterplot(data=df_coreset_centers, x="x", y="y", legend=False)

        # plt.scatter(points[:, 0], points[:, 1], cmap="g", label="Original")
        # plt.scatter(centers[:, 0], centers[:, 1], cmap="b", label="Original centers")
        # plt.scatter(coreset_points[:,0], coreset_points[:,1], cmap="r", label="Coreset")
        # plt.scatter(coreset_centers[:, 0], coreset_centers[:, 1], cmap="r", label="Coreset centers")
        tot_time = time.time() - start_time
        
        file_title = f"cluster_{mode}_k={k}_e={epsilon}_n={n}_coresetsize={len(coreset_points)}, time={tot_time:.1f}"
        plt.title(file_title)
        sns.set_style("white")
        file_name = path + "/" + file_title + ".png"
        plt.grid()
        plt.savefig(file_name)
        # plt.show()
        # plt.close()

def image_segmentation(image_array, mode="non-parallel" ,k=k, epsilon=0.1, show=False):
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image_array.reshape((-1, 3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    weights = np.array([1 for _, _ in enumerate(pixel_vals)]).reshape(-1, 1)

    initial_centers, indices = kmeanspp(pixel_vals, k, show=False)
    if mode=="non-parallel":
        coreset_points, coreset_weights = coreset_construction(pixel_vals, weights, initial_centers)
    elif mode=="parallel":
        coreset_points, coreset_weights = coreset_construction_parallel(pixel_vals)
    
    kmeans_coreset = KMeans(n_clusters=k, random_state=42).fit(coreset_points)
    coreset_centers = kmeans_coreset.cluster_centers_

    center_labels = np.argmin(euclidean_distances(pixel_vals, coreset_centers), 1)

    segmented_data = coreset_centers[center_labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(image_array.shape).astype(int)

    plt.imshow(segmented_image)
    plt.show()

def run_coreset_construction(i, points_weights, k=k, epsilon=0.1):
    points_weights = np.array(list(points_weights))[0] # Needed for parallelization
    d = points_weights.shape[1] - 1
    points, weights = points_weights[:,:d], points_weights[:,d:d+1]
    # print(f"points: {points.shape}, weights: {weights.shape}")
    centers, indices = kmeanspp(points, k, show=False)
    new_point, new_weight = coreset_construction(points, weights, centers, epsilon=epsilon)
    new_point_weight = np.concatenate([new_point, new_weight], axis=1)
    # print(f"new_point_weight: {new_point_weight.shape}")
    yield i, new_point_weight

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

    coreset = coreset[0] 
    d = coreset.shape[1] - 1
    S, S_weights = coreset[:,:d], coreset[:,d:d+1]
    return S, S_weights

def experiment_1():
    n_samples_list = [1000] #, 5000, 10000, 50000, 100000, 500000, 1000000]
    epsilon_list = [1e-3] #, 1e-2, 0.05, 1e-1, 0.5]
    k_list = [3] #, 4, 5, 6, 7, 8]
    k = 3
    PLOT_FOLDER = "images/"
    for n in n_samples_list:
        for epsilon in epsilon_list:
            for k in k_list:
                dl = Dataloader()
                coords, labels = dl.get_data("blob", k=k, blob_size=n, show=False)
                blob_clustering(coords, epsilon=epsilon, k=k, mode="parallel", show=True, path=PLOT_FOLDER)

def experiment_2():
    # TODO: Jonas will design this experiment for image segmentation
    pass
    
def experiment_3():
    # TODO: Anton will design this experiment for parallel vs non-parallel
    pass

def experiment_3():
    # TODO: Anton will design this experiment for coreset vs non-coreset (maybe on image segmantion) 
    pass

def main():
    experiment_1()


    # time_start = time.time()
    # dl = Dataloader()
    # coords, labels = dl.get_data("blob", blob_size=n, show=False)
    # blob_clustering(coords, k=k, mode="parallel" ,show=True)
    # time_end = time.time()
    # print(f"It took {time_end - time_start}")
    # d = coords.shape[1]
    # coreset, coreset_weight = coreset_construction_parallel(coords)

    # print(f"sum of all weights {coreset_weight.sum()} and the total sum of all points {coords.shape[0]}")
    # print(f"coords size: {coords.shape[0]}, coreset size: {coreset.shape[0]}")
    # plt.scatter(coords[:,0], coords[:,1], cmap="r", label="Original")
    # plt.scatter(coreset[:, 0], coreset[:, 1], cmap="r", label="Coreset")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.close()
    
    # coords, labels = dl.get_data("lena")
    # image_segmentation(coords, mode="non-parallel", k=k, epsilon=EPSILON)

    # for N_SAMPLES in N_SAMPLES_LIST:
    #     n = N_SAMPLES
    #     for EPSILON_SAMPLE in EPSILON_LIST:
    #         epsilon = EPSILON_SAMPLE
    #         for K_SAMPLE in K_LIST:
    #             k = K_SAMPLE
    #             coords, labels = dl.get_data("blob", blob_size=n, k=k, show=False)
    #             d = coords.shape[1]
    #
    #             weights = np.array([1 for _, _ in enumerate(coords)]).reshape(-1, 1)
    #
    #             point_weights = np.concatenate([coords, weights], axis=1)
    #             run_coreset_construction(point_weights, k=k, epsilon=epsilon, show=True)
    # rdd = sc.parallelize(point_weights, NUM_MACHINES)
    #
    # points = rdd.mapPartitionsWithIndex(run_coreset_construction, preservesPartitioning=True)
    # points_out = points.collect()
    # s, s_weight = np.zeros((NUM_MACHINES, d)), np.zeros((0, 1))
    #
    # for s_i, s_weight_i in range(point_weights):
    #     # print(f"s: {s}, s_weight: {s_weight} ")
    #     s = np.concatenate([s, s_i])
    #     s_weight = np.concatenate([s_weight, s_weight_i])
    # print("öool")

if __name__ == '__main__':
    main()