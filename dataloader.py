import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

LENA_IMAGE_PATH = r"data/lena_color.tiff"
BABOON_IMAGE_PATH = r"data/baboon.png"

class dataloader():
    def __init__(self):
        pass

    def get_data(self, data_name, show=False):
        if data_name == "lena":
            lena_img = Image.open(LENA_IMAGE_PATH)
            lena = np.array(lena_img)
            if show:
                plt.imshow(lena)
                plt.show()
            return lena

        elif data_name == "baboon":
            lena_img = Image.open(BABOON_IMAGE_PATH)
            lena = np.array(lena_img)
            if show:
                plt.imshow(lena)
                plt.show()
            return lena

def main():
    dl = dataloader()
    data = dl.get_data("baboon", show=True)

if __name__ == '__main__':
    main()