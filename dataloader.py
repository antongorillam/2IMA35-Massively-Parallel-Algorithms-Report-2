import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

LENA_IMAGE_PATH = r"data/lena_color.tiff"
BABOON_IMAGE_PATH = r"data/baboon.png"

class Dataloader():
    def __init__(self):
        pass

    def get_data(self, data_name, blob_size=None, k=3, show=False):
        if data_name=="lena":
            lena_img = Image.open(LENA_IMAGE_PATH)
            lena = np.array(lena_img)
            if show:
                plt.imshow(lena)
                plt.show()
            return lena, 0

        elif data_name=="baboon":
            baboon_img = Image.open(BABOON_IMAGE_PATH)
            baboon = np.array(baboon_img)
            if show:
                plt.imshow(baboon)
                plt.show()
            return baboon, 0

        elif data_name=="blob":
            # , center_box=(-20, 20)
            coords, labels = datasets.make_blobs(n_samples=blob_size, centers=k, n_features=2, cluster_std=2, center_box=(-30, 30), random_state=42)
            sns.scatterplot(coords[:,0], coords[:,1], labels)
            if show:
                plt.grid()
                plt.show()
            plt.close()
            return coords, labels
