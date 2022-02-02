import random
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from scipy.signal import convolve2d, medfilt2d


def add_salt_pepper_noise(image: np.ndarray, p: float) -> np.ndarray:
    """
    Adds Salt & Pepper noise to image.

    Input:
    image -- grayscale image array in the range [0..255]
    p -- float number <1 representing the proportion of pixels that will be noisy (0.5 for example)

    Output:
    sp_noise_image -- grayscale image in the range [0..255] (same size as image)

    Method:
    p/2 pixels are randomly chosen as 0 and p/2 as 255.
    """

    sp_noise_image = image.copy()

    # Picking pixels for which to apply salt and pepper noise
    # random.sample(range, n) - returns n random numbers in the given range without repetition
    random_pixels: np.ndarray = np.array(random.sample(range(image.size), int(p * image.size)))
    salt_pixels: np.ndarray = random_pixels[:len(random_pixels)//2]
    pepper_pixels: np.ndarray = random_pixels[len(random_pixels)//2:]

    salt_rows: np.ndarray = salt_pixels // image.shape[0]
    salt_columns: np.ndarray= salt_pixels % image.shape[1]

    pepper_rows: np.ndarray = pepper_pixels // image.shape[0]
    pepper_columns: np.ndarray = pepper_pixels % image.shape[1]

    sp_noise_image[salt_rows, salt_columns] = 255
    sp_noise_image[pepper_rows, pepper_columns] = 0

    return sp_noise_image


def add_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """
    Adds Gaussian noise to image (assuming mean of gaussian noise is zero).

    Input:
    image -- grayscale image array in the range [0..255]
    std -- the std to be used for Gaussian distribution of noise

    Output:
    gaussian_noise_image -- grayscale image in the range [0..255] (same size as image)

    Method:
    For every pixel adds a random value which is chosen from a Gaussian distribution of std 'std'.
    """

    gaussian_noise_image: np.ndarray = image.copy()

    # Creating a gaussian noise
    image_gaussian_mask: np.ndarray = np.random.normal(loc=0, scale=std, size=image.shape)

    gaussian_noise_image: np.ndarray = gaussian_noise_image + image_gaussian_mask

    return gaussian_noise_image


def clean_image_median_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """
    De-noising image using median filtering.

    Input:
    image -- a grayscale image array in the range [0..255]
    radius –- the radius of the filtering mask (Filtering mask is square, thus window at location r,c is coordinated:
              [r-radius : r+radius+1, c-radius : c+radius+1])

    Output:
    median_image -- grayscale image in the range [0..255] (same size as image)

    Method:
    Applying median filtering with neighborhood of radius 'radius'.

    Note:
     We're ignoring edges (as for starting "the loop" from 'radius' till 'shape[0] – radius', and same for columns)
    """

    kernel_size = 2 * radius + 1

    # Applying Median Filter on image
    median_image = medfilt2d(image, kernel_size)

    return median_image


def clean_image_mean_filter(image: np.ndarray, radius: int, mask_std: float) -> np.ndarray:
    """
    De-noises image using mean filtering.

    Input:
    image -- a grayscale image array in the range [0..255]
    radius –- the radius of the filtering mask (Filtering mask is square)
    mask_std -- the std of the Gaussian mask

    Output:
    gaussian_image -- grayscale image in the range [0..255] (same size as image)

    Method:
    Applying 2D Convolution on image with Gaussian filter.

    Note:
     Reminder,
      Gaussian filter is: 𝑒 ^{−(𝑥2 + 𝑦2) /2σ2} . Where x and y runs from '-radius' till 'radius'.
    """

    kernel_size = 2 * radius + 1

    y, x = np.indices((kernel_size, kernel_size)) - radius  # y - change in rows, x - change in columns
    gaussian_filter = np.exp(-(x ** 2 + y ** 2) / (2 * (mask_std ** 2)))
    gaussian_filter /= np.sum(gaussian_filter)

    gaussian_image = convolve2d(image, gaussian_filter, mode='same')

    return gaussian_image


def clean_image_bilateral_filter(image: np.ndarray, radius: int, std_spatial: float, std_intensity: float) -> np.ndarray:
    """
    This function applies Bilateral Filtering to the given image.
    Bilateral filtering replaces each pixel with a weighted average of its neighbors where the weights are determined
    according to the spatial and photometric (intensity) distances.

    Inputs:
    image -- grayscale image (array of values in [0..255])
    radius –- the radius of the neighborhood (neighborhood is square)
    std_spatial –- the std of the Gaussian function used for the spatial weights
    std_intensity -– the std of the Gaussian function used for the intensity weights

    Output:
    bilateral_image -- grayscale image (array of values in [0..255]) - the filtered image

    Method:
    Per pixel, we determine the local mask based on spatial and photometric weights. Normalizing the mask appropriately
    (image average should remain approx the same). Scanning the rows and columns of the image, but per each pixel we use
     matrix operations (not loops).

    For every pixel [i, j], we're building three masks:
    Window –- the image pixels in the neighborhood of the pixel [i,j]
    gaussian_intensity_mask -– gaussian mask based on intensity differences between pixel [i,j] and pixels in it’s
                               neighborhood [x,y]
    gaussian_distances_mask -- gaussian mask based on distances between pixel [i,j] and pixels in it’s
                               neighborhood [x,y]

    Note:
     𝑤𝑖𝑛𝑑𝑜𝑤 = 𝑖𝑚[𝑖−𝑟𝑎𝑑𝑖𝑢𝑠 : 𝑖+𝑟𝑎𝑑𝑖𝑢𝑠+1, 𝑗−𝑟𝑎𝑑𝑖𝑢𝑠 : 𝑗+𝑟𝑎𝑑𝑖𝑢𝑠+1]
     𝑔𝑖 = 𝑒 ^{−(𝑖𝑚[𝑥,𝑦] − 𝑖𝑚[𝑖,𝑗])**2 /2σ2}
     𝑔𝑠 = 𝑒 ^{(−(𝑥−𝑖)2 + (𝑦−𝑗)2) /2σ2}
        o Normalize gi, gs so that the sum of the elements in each one is 1.

    => 𝑛𝑒𝑤_𝑖𝑚𝑎𝑔𝑒[𝑖,𝑗] = 𝑠𝑢𝑚(𝑔𝑖 ∗ 𝑔𝑠 ∗ 𝑤𝑖𝑛𝑑𝑜𝑤) /𝑠𝑢𝑚(𝑔𝑖 ∗ 𝑔𝑠)
        o gi, gs, window: are all 2d of size (2radius+1)x(2radius+1).
          there is no need for loops to calculate them. there is only need for 2 loops to iterate over i and j for
          pixels. We're making sure we're dealing with floats in the calculation.
    """

    bilateral_image = image.copy()

    kernel_size = 2 * radius + 1
    rows = image.shape[0]
    columns = image.shape[1]

    y, x = np.indices((kernel_size, kernel_size)) - radius  # y - change in rows, x - change in columns
    gaussian_distances_mask: np.ndarray = np.exp(-(x ** 2 + y ** 2) / (2 * (std_spatial ** 2)))
    gaussian_distances_mask /= np.sum(gaussian_distances_mask)

    # Run on smaller image without the radius edges
    for i in range(radius, rows - radius):
        for j in range(radius, columns - radius):
            window = image[(i - radius):(i + radius + 1), (j - radius):(j + radius + 1)]

            gaussian_intensity_mask = np.power(np.e, -((window - image[i, j]) ** 2) / (2 * (std_intensity ** 2)))

            # Normalizing Gaussian masks
            gaussian_intensity_mask /= np.sum(gaussian_intensity_mask)

            bilateral_image[i, j] = np.divide(np.sum(gaussian_intensity_mask * gaussian_distances_mask * window),
                                              np.sum(gaussian_intensity_mask * gaussian_distances_mask))

    return bilateral_image
