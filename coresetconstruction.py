from cProfile import label
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

a = 1
n = 1000
N_SAMPLES_LIST = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
EPSILON_LIST = [1e-3, 1e-2, 0.05, 1e-1, 0.5]
K_LIST = [3, 4, 5, 6, 7, 8]
k = 3
NUM_MACHINES = 5
epsilon = 0.01
PLOT_FOLDER = "plots"

def coreset_construction(points, weights, centers, epsilon=epsilon):

    point_weights = dict()
    d = points.shape[1]

    z = np.log(n) * np.log(a * np.log(n))
    distances = euclidean_distances(points, centers)
    r = np.sqrt(
            np.sum(np.power(distances, 2))/(a * np.log(n) * n)
            )

    closest_centers = np.argmin(distances, 1)
    min_distances = np.min(distances, axis=1)
    # print(f"r: {r}")
    # I formulated point_weight s.t. its:
    #   key = cord of the center of the grid
    #   value = weight 

    # print(f"ITER {0} --------------------------------")
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
            point_weights[grid_position][1] += 1
        else: 
            point_weights[grid_position] = [point, weight]

    for j in range(1, int(z)+1):
        # print(f"ITER {j} --------------------------------")
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
                point_weights[grid_position][1] += 1
            else: 
                point_weights[grid_position] = [point, weight]
            break
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

def run_coreset_construction_nonparallel(points_weights, k=k, epsilon=epsilon, show=False):
    points, weights = points_weights[:,:2], points_weights[:,2:3]
    centers, indices = kmeanspp(points, k, show=False)
    coreset_points, coreset_weights = coreset_construction(points, weights, centers)
    kmeans_normal = KMeans(n_clusters=k, random_state=42).fit(points)
    kmeans_coreset = KMeans(n_clusters=k, random_state=42).fit(coreset_points)

    normal_centers = kmeans_normal.cluster_centers_

    coreset_centers = kmeans_coreset.cluster_centers_
    coreset_center_labels = kmeans_coreset.labels_

    coreset_dist = euclidean_distances(coreset_points, centers).min(axis=1)
    coreset_cost = np.sum(np.power(coreset_dist, 2) * coreset_weights)

    new_coreset_distances = euclidean_distances(coreset_points, coreset_centers)
    new_closest_centers = np.argmin(new_coreset_distances, 1)

    if show:
        point_data = {
            "x": points[:, 0],
            "y": points[:, 1],
            "label": kmeans_normal.labels_
        }

        center_data = {
            "x": normal_centers[:, 0],
            "y": normal_centers[:, 1]
        }

        coreset_point_data = {
            "x": coreset_points[:, 0],
            "y": coreset_points[:, 1],
            "label": coreset_center_labels
        }

        coreset_center_data = {
            "x": coreset_centers[:, 0],
            "y": coreset_centers[:, 1],
            "label": [i for i in range(len(coreset_centers))]
        }

        df_normal = pd.DataFrame(point_data)
        df_centers = pd.DataFrame(center_data)
        df_coreset = pd.DataFrame(coreset_point_data)
        df_coreset_centers = pd.DataFrame(coreset_center_data)

        # plt.figure()
        # plt.grid()
        # palette1 = sns.dark_palette("seagreen", k)
        # palette2 = sns.color_palette("deep", k)
        # sns.scatterplot(data=df_normal, x="x", y="y", hue="label", palette=palette1, legend=False)
        # sns.scatterplot(data=df_coreset, x="x", y="y", hue="label", palette=palette1)
        # sns.scatterplot(data=df_centers, x="x", y="y", palette="red")
        # sns.scatterplot(data=df_coreset_centers, x="x", y="y", legend=False)

        # plt.scatter(points[:, 0], points[:, 1], cmap="g", label="Original")
        # plt.scatter(centers[:, 0], centers[:, 1], cmap="b", label="Original centers")
        # plt.scatter(coreset_points[:,0], coreset_points[:,1], cmap="r", label="Coreset")
        # plt.scatter(coreset_centers[:, 0], coreset_centers[:, 1], cmap="r", label="Coreset centers")
        # title = f"Coreset Points with k={k}_e={epsilon}_n={n} - Non-parallel"
        # plt.title(title)
        #
        # file_title = f"Coreset_points_non_parallel_k={k}_e={epsilon}_n={n}"
        # file_name = PLOT_FOLDER + "/" + file_title + ".png"
        # plt.grid()
        #
        # plt.savefig(file_name)
        # plt.show()
        # plt.close()

def image_segmentation_nonparallel(image_array, k=k, epsilon=epsilon, show=False):
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image_array.reshape((-1, 3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    weights = np.array([1 for _, _ in enumerate(pixel_vals)]).reshape(-1, 1)

    initial_centers, indices = kmeanspp(pixel_vals, k, show=False)
    coreset_points, coreset_weights = coreset_construction(pixel_vals, weights, initial_centers)
    kmeans_coreset = KMeans(n_clusters=k, random_state=42).fit(coreset_points)
    coreset_centers = kmeans_coreset.cluster_centers_

    center_labels = np.argmin(euclidean_distances(pixel_vals, coreset_centers), 1)

    segmented_data = coreset_centers[center_labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(image_array.shape).astype(int)

    plt.imshow(segmented_image)
    plt.show()



def run_coreset_construction(split_index, points_weights, k=k, epsilon=0.05):
    
    points_weights = np.array(list(points_weights)) # Needed for parallelization
    points, weights = points_weights[:,:2], points_weights[:,2:3]
    centers, indices = kmeanspp(points, k, show=False)
    new_point, new_weight = coreset_construction(points, weights, centers)
    new_point_weight = np.concatenate([new_point, new_weight], axis=1)
    # print(f"new point and weight: {new_point.shape}, {new_weight.shape}")
    # print(f"new_point_weight: {new_point_weight}")
    # print(f"new_point_weight[:,2].sum(): {new_point_weight[:,2].sum()}")
    print(f"new_point_weight {new_point_weight.shape}")
    print(f"split_index {split_index}")
    yield split_index, new_point_weight

def main():
    dl = Dataloader()

    coords, labels = dl.get_data("baboon")
    image_segmentation_nonparallel(coords, k=k, epsilon=epsilon)
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
    #             run_coreset_construction_nonparallel(point_weights, k=k, epsilon=epsilon, show=True)
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