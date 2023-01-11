import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql.session import SparkSession
spark = SparkSession.builder.appName("DFTest").getOrCreate()
sc = spark.sparkContext

from coresetconstruction import coreset_construction
from kmeanspp import kmeanspp
from dataloader import Dataloader

# def main():
#     dl = Dataloader()
#     coords, k = dl.get_data("blob", blob_size=10000 ,show=False)
#     centers, indices = kplusplus(coords, k, False)
#     # sns.scatterplot(coords[:, 0], coords[:, 1], k)
#     # plt.grid()
#     # plt.show()
#     coords_rdd = sc.parallelize(coords, 100)
#     sqrt = coords_rdd.map(lambda x : x[0] + x[1])
#     print("klol")

def main():
    dl = Dataloader()
    coords, k = dl.get_data("blob", blob_size=1000 ,show=False)
    rdd = sc.parallelize(coords, 2)
    centers = rdd.mapPartitions(kmeanspp)
    centers, coords = centers.collect()
    # centers, indices = kplusplus(coords, k, show=False)
    # point_weights = coreset_construction(coords, centers)
    print("asd")

if __name__ == '__main__':
    main()