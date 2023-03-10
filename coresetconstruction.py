from dataloader import Dataloader
from kmeanspp import kmeanspp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from pyspark.sql.session import SparkSession
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

spark = SparkSession.builder.appName("DFTest").getOrCreate()
sc = spark.sparkContext

a = 1
n = 50000
N_SAMPLES_LIST = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
EPSILON_LIST = [1e-3, 1e-2, 0.05, 1e-1, 0.5]
K_LIST = [3, 4, 5, 6, 7, 8]
NUM_MACHINES = [4, 2]

k = 3
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

    return S, S_weights

def blob_clustering(points_weights, k=k, epsilon=0.1, mode="non-parallel", show=False, path=None):
    
    start_time = time.time()
    points, weights = points_weights[:,:2], points_weights[:,2:3]
    initial_centers, indices = kmeanspp(points, k, show=False)
    if mode=="non-parallel":
        coreset_points, coreset_weights = coreset_construction(points, weights, initial_centers)
    elif mode=="parallel":
        coreset_points, coreset_weights = coreset_construction_parallel(points)

    kmeans_coreset = KMeans(n_clusters=k, random_state=42).fit(coreset_points)
    final_centers = kmeans_coreset.cluster_centers_

    center_labels = np.argmin(euclidean_distances(points, final_centers), 1)

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
        palette1 = sns.dark_palette("seagreen", k)
        sns.scatterplot(data=df_normal, x="x", y="y", hue="label", palette=palette1, legend=False)
        tot_time = time.time() - start_time
        
        file_title = f"cluster_{mode}_k={k}_e={epsilon}_n={n}_coresetsize={len(coreset_points)}, time={tot_time:.1f}"
        plt.title(file_title)
        sns.set_style("white")
        file_name = path + "/" + file_title + ".png"
        plt.grid()
        plt.savefig(file_name)

        return tot_time, len(coreset_points)

def image_segmentation(image_array, mode="non-parallel" ,k=k, epsilon=0.1, show=False, path=None):
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    time_start = time.time()
    pixel_vals = image_array.reshape((-1, 3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    weights = np.array([1 for _, _ in enumerate(pixel_vals)]).reshape(-1, 1)

    initial_centers, indices = kmeanspp(pixel_vals, k, show=False)
    if mode=="non-parallel":
        coreset_points, coreset_weights = coreset_construction(pixel_vals, weights, initial_centers)
    elif mode=="parallel":
        coreset_points, coreset_weights = coreset_construction_parallel(pixel_vals)
    elif mode=="no-coreset":
        coreset_points = pixel_vals
        coreset_weights = None
    else:
        raise f"Mode: {mode} is not valid."

    kmeans_coreset = KMeans(n_clusters=k, random_state=42).fit(coreset_points)
    coreset_centers = kmeans_coreset.cluster_centers_

    center_labels = np.argmin(euclidean_distances(pixel_vals, coreset_centers), 1)

    segmented_data = coreset_centers[center_labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(image_array.shape).astype(int)
    total_time = time.time() - time_start
    file_title = f"epsilon={epsilon}_k={k}_mode={mode}_totaltime={total_time:.1f}"
    plt.title(file_title)
    plt.imshow(segmented_image)
    plt.savefig(path + file_title + ".png")
    return total_time, coreset_points, coreset_weights, coreset_centers

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
    mapped_points = np.array(index_rdd.collect(), dtype=object)

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

    rdd = sc.parallelize(coreset, 1)
    coreset = rdd.mapPartitionsWithIndex(run_coreset_construction, preservesPartitioning=True).collect()
    coreset = coreset[0][1] 
    d = coreset.shape[1] - 1
    S, S_weights = coreset[:,:d], coreset[:,d:d+1]

def experiment_1():
    image_list = ["lena", "baboon"]
    epsilon_list = [3e-1]
    k_list = [3, 5, 10, 30, 50]

    total_runs = len(image_list) * len(epsilon_list) * len(k_list)
    iter = 0
    df = pd.DataFrame(
        columns=["image", "epsilon", "k", "mode", "execution time", "number of machines", "cost"]
    )
    PLOT_FOLDER = "images/experiment_1/"
    start_time = time.time()
    for image in image_list:
        for epsilon in epsilon_list:
            for k in k_list:
                iter += 1
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"run {iter}/{total_runs}, time elapsed {time_elapsed}")
                dl = Dataloader()
                image_array, _ = dl.get_data(image, k=k, show=False)
                tot_time_parallel, coreset_points_parallel, coreset_weights_parallel, coreset_centers_parallel  = image_segmentation(image_array, mode="parallel" ,k=k,
                                                                     epsilon=epsilon, show=False, path=PLOT_FOLDER)
                tot_time_nonparallel, coreset_points_nonparallel, coreset_weights_nonparallel, coreset_centers_nonparallel = image_segmentation(image_array, mode="non-parallel" ,k=k,
                                                                     epsilon=epsilon, show=False, path=PLOT_FOLDER)
                # tot_time_normal, coreset_points_normal, coreset_weights_normal, coreset_centers_normal = image_segmentation(image_array, mode="no-coreset", k=k,
                #                                              epsilon=epsilon, show=False, path=PLOT_FOLDER)

                coreset_parallel_dist = euclidean_distances(coreset_points_parallel, coreset_centers_parallel).min(axis=1)
                coreset_parallel_cost = np.sum(np.power(coreset_parallel_dist, 2) * coreset_weights_parallel)

                coreset_nonparallel_dist = euclidean_distances(coreset_points_nonparallel, coreset_centers_nonparallel).min(axis=1)
                coreset_nonparallel_cost = np.sum(np.power(coreset_nonparallel_dist, 2) * coreset_weights_nonparallel)


                # parallel_accuracy = np.sum(coreset_parallel_labels == ground_truth_labels) / len(ground_truth_labels)
                # nonparallel_accuracy = np.sum(coreset_nonparallel_labels == ground_truth_labels) / len(ground_truth_labels)

                df = df.append(
                    {"image": image, "epsilon": epsilon, "k": k, "mode": "parallel",
                     "execution time": tot_time_parallel, "number of machines": NUM_MACHINES[0], "cost": coreset_parallel_cost},
                    ignore_index=True)
                df = df.append(
                    {"image": image, "epsilon": epsilon, "k": k, "mode": "non-parallel",
                     "execution time": tot_time_nonparallel, "number of machines": NUM_MACHINES[0], "cost": coreset_nonparallel_cost},
                    ignore_index=True)

    df.to_csv(PLOT_FOLDER + "performance_data_costs.csv")
    pass
    
def experiment_2():
    n_samples_list = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    epsilon_list = [1e-3, 1e-2, 1e-1]
    k_list = [15]
    total_runs = len(n_samples_list) * len(epsilon_list) * len(k_list)
    iter = 0
    df = pd.DataFrame(
        columns=["n_samples", "epsilon", "k", "mode", "execution time", "number of machines"]
    )
    PLOT_FOLDER = "images/experiment_2/"
    start_time = time.time()
    for n in n_samples_list:
        for epsilon in epsilon_list:
            for k in k_list:
                iter += 1
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"run {iter}/{total_runs}, time elapsed {time_elapsed}")
                dl = Dataloader()
                coords, _ = dl.get_data("blob", k=k, blob_size=n, show=False)
                tot_time_parallel, coreset_size = blob_clustering(coords, epsilon=epsilon, k=k, mode="parallel", show=True, path=PLOT_FOLDER)
                tot_time_normal, _ = blob_clustering(coords, epsilon=epsilon, k=k, mode="non-parallel", show=True, path=PLOT_FOLDER)
                df = df.append(
                    {"n_samples":n ,"epsilon":epsilon, "k":k, "coreset size":coreset_size, "mode":"parallel", "execution time":tot_time_parallel, "number of machines":NUM_MACHINES[0]},
                    ignore_index=True)
                df = df.append(
                    {"n_samples":n ,"epsilon":epsilon, "k":k, "coreset size":None , "mode":"non-parallel", "execution time":tot_time_normal, "number of machines":NUM_MACHINES[0]},
                    ignore_index=True)

    df.to_csv(PLOT_FOLDER + "performance_data.csv")

def experiment_3():
    # Experiment for coreset vs non-coreset (maybe on image segmantion) 
    epsilon_list = [1e-1, 3e-1, 5e-1, 7e-1, 9e-1]
    k_list = [30]
    DATASET_NAME = ["lena", "baboon"]
    total_runs = len(DATASET_NAME) * len(epsilon_list) * len(k_list)
    iter = 0
    
    df = pd.DataFrame(
        columns=["dataset", "epsilon", "k", "mode", "execution time", "number of machines"]
    )

    start_time = time.time()
    for dataset in DATASET_NAME:
        PLOT_FOLDER = f"images/experiment_3/"
        for epsilon in epsilon_list:
            for k in k_list:
                iter += 1
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"run {iter}/{total_runs}, time elapsed {time_elapsed}, k {k}, epsilon {epsilon}, dataset {dataset}")
                
                dl = Dataloader()
                coords, _ = dl.get_data(dataset, k=k, blob_size=n, show=False)
                tot_time_parallel = image_segmentation(coords, epsilon=epsilon, k=k, mode="parallel", show=True, path=PLOT_FOLDER + f"/{dataset}_")
                tot_time_normal = image_segmentation(coords, epsilon=epsilon, k=k, mode="no-coreset", show=True, path=PLOT_FOLDER + f"/{dataset}_")
                df = df.append(
                    {"dataset":dataset, "epsilon":epsilon, "k":k, "mode":"parallel", "execution time":tot_time_parallel, "number of machines":NUM_MACHINES[0]},
                    ignore_index=True)
                df = df.append(
                    {"dataset":dataset, "epsilon":epsilon, "k":k, "mode":"no-coreset", "execution time":tot_time_normal, "number of machines":NUM_MACHINES[0]},
                    ignore_index=True)

    df.to_csv(PLOT_FOLDER + "higher_epsilon_performance_data.csv")

def main():
    # experiment_1()
    # experiment_2()
    # experiment_3()

if __name__ == '__main__':
    main()