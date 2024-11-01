# from https://github.com/Crazz-Zaac/mathematical-image-processing/blob/master/exercise_solutions/ex1/contrast_enhancement.py
import numpy as np
import matplotlib.pyplot as plt


def my_hist(I: np.ndarray, nbins: int) -> np.ndarray:
    """
    Computes histogram of and Image I with number of nbins.
    """
    output_hist, bins = np.histogram(I.flatten(), bins=nbins, range=(0, 255))
    return output_histl


def plot_hist(I: np.ndarray, nbins: int) -> None:
    """
    Plots the histogram of an Image I with number of nbins.
    """
    output_hist, bins = my_hist(I, nbins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.imshow(I)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.bar(bins[:-1], output_hist, width=3)
    ax2.set_title("Histogram of pixels distribution")
    ax2.set_xlabel("pixel values")
    ax2.set_ylabel("pixel counts")

    plt.show()


def hist_linear(I: np.ndarray, range_min: int, range_max: int) -> np.ndarray:
    """
    Linear transformation of an image I to a range of [range_min, range_max].
    """
    image_array = np.array(I)
    min_v = np.min(I)
    max_v = np.max(I)

    image_transformed = ((image_array - min_v) / (max_v - min_v)) * \
        (range_max - range_min) + range_min

    return np.clip(image_transformed, range_min, range_max)


def hist_gamma(I: np.ndarray, gamma: float) -> np.ndarray:
    """
    Non-linear transformation of an image I with gamma value.
    """
    image_array = np.array(I)
    I_gamma = 255 * np.power(image_array / 255, gamma)

    return np.clip(I_gamma, 0, 255).astype(np.uint8)


def hist_eq(I: np.ndarray) -> np.ndarray:
    """
    A function to perform a histogram equalization
    """
    image_array = np.array(I)
    hist, bins = np.histogram(
        image_array, bins=1000, range=(0, 255), density=True)

    # cumulative sum of histogram
    cdf = hist.cumsum()

    # normalized histogram
    normalized_cdf = 255 * cdf / cdf[-1]
    I_equalized = np.interp(image_array.flatten(
    ), bins[:-1], normalized_cdf).reshape(image_array.shape)

    return I_equalized.astype(np.uint8)