import matplotlib.pyplot as plt
from numpy.typing import NDArray

eurosat_train_means = [1353.439, 1117.253, 1042.253, 947.128, 1199.404, 2002.936, 2373.488, 2300.642, 732.159, 12.113, 1820.932, 1119.173, 2598.82]
eurosat_train_std_devs = [65.571, 154.376, 188.262, 278.926, 228.244, 355.633, 454.901, 530.549, 98.716, 1.187, 378.496, 304.439, 501.747]

def plot_eurosat_image(image: NDArray, title: str):
    band_names = ["coastal_blue", "blue", "green", "red", "red_edge1", "red_edge2", "red_edge3", "nir", "narrow_nir", "water_vapour", "cirrus", "swir1", "swir2"]
    image = (image - image.min()) / (image.max() - image.min())
    _, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (10, 10))
    for idx, ax in enumerate(axes.ravel()):
        if idx < image.shape[0]:
            ax.imshow(image[idx], cmap = "grey")
            ax.set_xlabel(f"#{idx+1} : {band_names[idx]}")
        ax.tick_params(axis = "both", which = "both", 
                        bottom = False, top = False, 
                        left = False, right = False,
                        labeltop = False, labelbottom = False, 
                        labelleft = False, labelright = False)
    plt.suptitle(title);
    plt.tight_layout();
    plt.show();