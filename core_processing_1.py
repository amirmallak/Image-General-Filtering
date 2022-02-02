import numpy as np
import matplotlib.pyplot as plt

from typing import List
from cv2 import cv2


def contrast_enhancement(image: np.ndarray, image_desired_range: List[int]) -> (np, float, float):
    """
    This function maps an image to new range of gray values. It can be used to enhance contrast.
    Mapping is linear in the form 𝑔𝑛𝑒𝑤 = 𝑎∗ 𝑔𝑜𝑙𝑑 + 𝑏.

    Input:
    image -- a grayscale image in the range [0..255]
    image_desired_range -– range of gray values in the form [minVal, maxVal]

    Return:
    enhanced_image –- the new grayscale image (same size as image) – a 2D numpy array
    scale, bias -- the parameters of the Tone Mapping that performs the mapping

    Method:
    Maps image such that the new image enhanced_image has values in the range given as input.
    Mapping is linear and is in the form 𝑔𝑛𝑒𝑤 = 𝑎∗ 𝑔𝑜𝑙𝑑 + 𝑏. Function returns mapped image as well as
    the parameters scale and bias.

    Example usage:
    enhanced_image, scale, bias = contrastEnhance(image, [10, 250]).
    """

    image_min_value = image.min(axis=None, initial=None)
    image_max_value = image.max(axis=None, initial=None)
    image_range = np.abs(image_max_value - image_min_value)
    range_of_desired_range = image_desired_range[1] - image_desired_range[0]
    scale = range_of_desired_range / image_range
    bias = image_desired_range[1] - (scale * image_max_value)
    enhanced_image = (scale * image + bias).round()

    return enhanced_image, scale, bias


def show_image_mapping(old_range: List[int], scale_factor: float, bias_factor: float) -> None:
    image_min_range = np.min(old_range)
    image_max_range = np.max(old_range)
    x = np.arange(image_min_range, image_max_range + 1, dtype=np.float)
    y = scale_factor * x + bias_factor
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')


def image_resizing(image_1: np.ndarray, image_2: np.ndarray) -> (np.ndarray, np.ndarray):
    # Resizing the bigger image to the size of the smaller one
    bigger_image = image_1 if (image_1.size > image_2.size) else image_2
    smaller_image = image_1 if (image_1.size <= image_2.size) else image_2
    smaller_image_dim = (smaller_image.shape[1], smaller_image.shape[0])
    resized_bigger_image = cv2.resize(bigger_image, smaller_image_dim, interpolation=cv2.INTER_CUBIC)

    return resized_bigger_image, smaller_image


def minkowski_2_distance(image_1: np.ndarray, image_2: np.ndarray) -> float:
    """
    This function measures the Minkowski distance between (histograms of) two images. Using p=2.

    Input:
    image_1, image_2 –- 2D numpy matrices. These are grayscale images with values in the range [0..255]

    Return:
    minkowski_2_dist -- the distance value

    Method:
    Using np.histogram(image1, bins = 256, range=(0,255)), using numpy functions: sum, power, casting to float.

    Note:
    image_1 and image_2 need not be same size.
    """

    resized_bigger_image, smaller_image = image_resizing(image_1, image_2)

    # Calculating Minkowski p=2 Dist
    p = 2
    histogram_1 = np.histogram(resized_bigger_image, bins=256, range=(0, 255))
    histogram_1 = histogram_1[0] / resized_bigger_image.size
    histogram_2 = np.histogram(smaller_image, bins=256, range=(0, 255))
    histogram_2 = histogram_2[0] / smaller_image.size
    image_difference = np.subtract(histogram_1, histogram_2)
    mean_square_error = np.power(image_difference, p)
    sum_of_histograms = np.sum(mean_square_error)
    minkowski_2_dist = np.power(sum_of_histograms, 1 / p)

    return minkowski_2_dist


def mean_square_distance(image_1: np.ndarray, image_2: np.ndarray) -> float:
    """
    This function measures the mean square distance between two images.

    Input:
    image_1, image_2 -– Matrices of grayscale images of the same size in range [0..255]

    Return:
    mean_square_dist -- The distance value

    Method:
    Distance is the mean of the squared distances between corresponding pixels of the 2 IMAGES (not histograms).

    Note:
    Make sure values are float!
    """

    mean_square_dist = np.power(np.sum(np.power(np.subtract(image_1.astype(float), image_2.astype(float)), 2)), 5e-1) \
                       / image_1.size

    return mean_square_dist


def slice_matrix(image: np.ndarray) -> np.ndarray:
    """
    This function builds the Slice Matrix, a binary valued matrix of size (numPixel x 256).

    Input:
    image -- a 2D grayscale image matrix in the range [0..255]

    Return:
    slices -– a binary valued matrix (2D numpy array) of size (numPixel x 256),
    where (numPixel is the number of pixels in image)

    Method:
    The slices matrix contains the slices along the columns.
    Column i is associated with gray value i. The column i contains 1 in entry k if image(:) has value i at index k.
    That is: slices(k,i) == 1 if image(k) = i.
    We're looping through only one loop (looping over 256 gray levels), instead of looping on the image pixels to create
    Boolean masks.
    """

    gray_color_range = 256
    # Matrix size: (Number_of_Image_Pixels x 256)
    slices = np.ndarray(shape=(image.size, gray_color_range), dtype=float)
    flattened_image = image.flatten()
    for gray_color in range(gray_color_range):
        image_boolean_mask = (flattened_image == gray_color)
        image_binary_mask = image_boolean_mask * 1
        slices[:, gray_color] = image_binary_mask

    return slices


def slt_map(image_1: np.ndarray, image_2: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Tone Maps image image_1 to be as similar to image im2, and outputs the mapped image as well as the tone mapping
    function.

    Input:
    image_1, image_2 -- a grayscale images of the same size in the range [0..255]

    return:
    mapped_image_1 –- the new grayscale image (same size as image_1)
    tone_mapping_vector –- a tone map = a 1x256 vector defining a mapping

    Method:
    Uses the SL matrix for image_1 calculated using sliceMat. For each slice (column) i, finds the gray values in
    image_2 that correspond to values of 1 in the slice.
    These gray values are averaged to create the value i mapped_image_1 to which gray scale i is mapped:
    tone_mapping_vector[i] = i mapped_image_1.
    Image image_1 is tone mapped using tone_mapping_vector to create new image mapped_image_1.
    """

    # Binary masked image with Shape (number_of_image_pixels x 256)
    image_1_sliced_matrix: np.ndarray = slice_matrix(image_1)
    image_1_number_of_appearances = np.sum(image_1_sliced_matrix, axis=0)  # How many appearances from each gray color
    image_2_flatten = image_2.flatten()
    mapping_color_vector = np.matmul(image_2_flatten, image_1_sliced_matrix)  # Vector of shape (1 x 256)
    mapping_color_vector /= image_1_number_of_appearances
    tone_mapping_vector = np.nan_to_num(mapping_color_vector)

    mapped_image_1 = mapping_image(image_1, tone_mapping_vector)

    return mapped_image_1, tone_mapping_vector


def mapping_image(image: np.ndarray, tone_mapping_vector: np.ndarray) -> np.ndarray:
    """
    Maps image grayscale according to given toneMap. Returns new image (uses slice_matrix function).

    Input:
    image -- a grayscale image matrix in the range [0..255]
    tone_mapping_vector – tone map = a 1x256 (numpy array) defining a tone mapping

    return:
    tone_mapped_image –- the new grayscale image (same size as image)

    Method:
    Map gray value i in image to new value given in tone_mapping_vector(i).
    uses np.reshape to return tone_mapped_image of same size as image.
    """

    sliced_image: np.ndarray = slice_matrix(image)  # Shape (number_of_image_pixels x 256)
    flattened_tone_mapped_image = np.matmul(sliced_image, tone_mapping_vector.T)  # Shape (number_of_image_pixels x 1)
    tone_mapped_image = flattened_tone_mapped_image.T.reshape(image.shape)

    return tone_mapped_image


def slt_negative(image: np.ndarray) -> np.ndarray:
    """
    Maps image to it’s negative (uses mapping_image function)

    Input:
    image -- a grayscale image matrix in the range [0..255]

    Return:
    negative_mapped_image -– the negative grayscale image

    Method:
    Using slt slicing and mapping, calculates negative image.
    Uses np.arange(r) to create an array of values [0,1,2,3,…, r]
    """

    tone_mapping_vector = 255 - np.arange(256)  # Tone mapping negative image vector. Shape (1 x 256)
    negative_mapped_image = mapping_image(image, tone_mapping_vector)

    return negative_mapped_image


def slt_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Performs thresholding on image (uses mapImage function)

    Input:
    image -- a grayscale image matrix in the range [0..255]
    threshold –- threshold value

    Return:
    threshold_image –- binary image (2D numpy array)

    Method:
    Using slt slicing and mapping , performs thresholding
    (thresholding = each graylevel > thresh turns to 255 and each graylevel <= thresh turn to 0).
    Uses np.arange(r) to create an array of values [0,1,2,3,…, r]
    """

    gray_scale_vector = np.arange(256)
    boolean_tone_mapping_vector = gray_scale_vector > threshold
    tone_mapping_vector = 255 * boolean_tone_mapping_vector
    threshold_image = mapping_image(image, tone_mapping_vector)

    return threshold_image
