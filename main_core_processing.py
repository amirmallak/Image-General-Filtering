import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from scipy.signal import convolve2d
from .core_processing_1 import contrast_enhancement
from .core_processing_3 import clean_image_median_filter
from .core_processing_2 import find_affine_transform, find_projective_transform, map_image, get_image_points


def clean_image_1(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'baby' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    Swapping on the input image (which contains 3 images) and cleaning the salt-pepper noise by applying a median
    filter. After, picking the boundary of one of the sub-images inside the input image and finding the transformation
    matrix - T (an affine or projective transformation) and applying it on the above sub-image to a new image frame.

    """

    # image = clean_image_median_filter(image, 1)
    
    radius = 1
    median_image = image.copy()
    image_row = image.shape[0]
    image_col = image.shape[1]

    # Applying a median filter
    for row in range(radius, image_row - radius):
        for col in range(radius, image_col - radius):
            if image[row, col] == 0 or image[row, col] == 255:
                radius_data = image[row - radius: row + radius + 1, col - radius: col + radius + 1]
                new_val = np.median(radius_data)
                median_image[row, col] = new_val

    image = median_image

    # Picking the sub-image (inside the input image) frame points
    points_path = 'image_1_points'
    # get_image_points(image, points_path, 4)
    # image_1_points = np.array([[20, 6], [20, 111], [130, 6], [130, 111]])
    # np.save(points_path, image_1_points)

    image_1_points = np.load(f'{points_path}.npy')

    # Picking points which represents the destination frame to which the transformation of the sub-image will be
    # transformed to
    destination_points = np.array([[0, 0], [0, image.shape[1]], [image.shape[0], 0], [image.shape[0], image.shape[1]]])

    # Calculating the transformation matrix
    # T = find_affine_transform(image_1_points, destination_points)
    T = find_projective_transform(image_1_points, destination_points)

    # Mapping the image to the new coordinates
    clean_image = map_image(image, T, image.shape)

    return clean_image


def clean_image_2(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'wind mil' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    The 'wind mil' image has stripes (with a fixed frequency) in a certain degree on it. The algorithm consists in
    applying a fourier transformation on the image, and since the stripes are at a fixed frequency, we'll find this
    frequency in the fourier domain at a certain point (u,v) represented as a two-symmetric points with high amplitude.
    We can eliminate those points by placing a new values at these specific fourier-domain point locations as 0.

    """

    # Applying a fourier transformation
    image_fourier = np.fft.fft2(image)
    image_fourier_shift = np.fft.fftshift(image_fourier)

    # Placing a zero value in fourier domain at those specific points (which representing a fixed-occurring frequency
    # at the image)
    image_fourier_shift[124, 100] = 0  # [124,100] visually found
    image_fourier_shift[132, 156] = 0  # [132,156] visually found

    # Applying an inverse fourier transformation (back to the image domain)
    clean_image = abs(np.fft.ifft2(image_fourier_shift))
    
    return clean_image


def clean_image_3(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'watermelon' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    The image 'watermelon' is presented as a low frequency image. Hence, we should enhance it.
    In order to do so cleanly (without ringing - a by-product of an ideal filter (whether high or low)), we'll create
    a gaussian mask (a not ideal high-pass filter) and apply it (multiply it) in the frequency domain on the fourier
    transform of the input image

    """

    image_fourier = np.fft.fft2(image)
    image_fourier_shift = np.fft.fftshift(image_fourier)

    # A numpy array containing the center values of the input image. Shape - (1x2)
    center = np.ceil(np.array(image.shape) / 2).astype('int')
    
    # creating a gaussian mask - sharpening without ringing
    r = 50
    std = 10
    mask = np.zeros([r * 2 + 1, r * 2 + 1])
    for row in range(r * 2 + 1):
        for col in range(r * 2 + 1):

            # Creating a low-pass filter (after, we'll turn it to a high pass)
            mask[row, col] = (1 / np.sqrt(2 * np.pi * std**2)) * (np.exp(-((row-r)**2 + (col-r)**2) / (2 * std**2)))

    # Creating a gaussian mask (in the frequency domain, and as the same shape as the input image) which at it's
    # boundaries (and a little bit inside) contains ones (maintaining the high frequencies), and at it's center
    # containing a gaussian mask such that at the center of this gaussian mask relies the lowest value within the mask!
    # -> (1 - mask). and at it's boundaries relies the highest (giving more wight to the higher frequencies out of the
    # lowest ones), and outside of this gaussian mask there're ones from the original 'mask_image'
    mask_image = np.ones(image.shape)

    # In order to give low values to the low frequencies
    # Creating a high-pass filter
    mask_image[(center[0] - r):(center[0] + r+1), (center[1] - r):(center[1] + r+1)] = (1 - mask)

    # decrease low frequencies amplitude (in the frequency domain)
    image_fourier_shift = (image_fourier_shift * mask_image)
    image_inverse = np.fft.ifft2(image_fourier_shift)
    clean_image = abs(image_inverse)
    
    # Enhancing the high frequencies
    clean_image, _, _ = contrast_enhancement(clean_image, [0, 255])
    
    return clean_image


def clean_image_4(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'umbrella' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    The 'umbrella' image is composed of the original umbrella image plus a time-domain shifted original umbrella image -
    I + I * delta(x - x_0, y - y_0) = I * [delta + delta(x - x_0, y - y_0)].

    We can find the [x_0, y_0] values by picking them up straight forward from the 'umbrella' image.
    After, we can calculate the fourier transform of the 'umbrella' image (F{I + I * delta(x - x_0, y - y_0)}), and
    divide it by the fourier transform of the shift portion (F{delta + delta(x - x_0, y - y_0)}).

    For the clean image, we'll apply an inverse fft at the divided image.

    """

    image_fourier = np.fft.fft2(image)
    # image_fourier_shift = np.fft.fftshift(image_fourier)

    # Creating a delta function
    delta = np.zeros(image.shape)
    delta[0, 0] = 1

    # Calculating the fourier transform of a delta function
    delta_fourier = np.fft.fft2(delta)

    # The points in the image where the shift occurs (those are the x_0 and y_0 values which the delta function has
    # shifted the original images by)
    shift = np.array([4, 79])

    # Creating a shifted delta function
    delta_shift = np.zeros(image.shape)
    delta_shift[shift[0], shift[1]] = 1

    # Calculating the fourier transform of the shifted delta function
    delta_shift_fourier = np.fft.fft2(delta_shift)

    # Calculating and dividing the fourier transforms of the input image (in frequency domain -
    # [image + image * exp(2 * pi * j * (u + v))] ), with the corresponding fourier transform of the correction needed
    # (in frequency domain - {exp(2 * pi * j * (u + v))} -> which in time domain (F-1{exp(2 * pi * j * (u + v))}) is
    # [delta(x, y) + delta(x - x_0, y - y_0)] -> the [delta + delta_shift] we've created earlier).
    image_fourier_shift = np.fft.fftshift(image_fourier / (delta_shift_fourier + delta_fourier) / 0.5)
    image_fourier_shift = np.nan_to_num(image_fourier_shift, nan=0)

    # Taking a threshold for better results
    image_fourier_shift[np.where(np.log(abs(image_fourier_shift)) > 35)] = 0

    image_inverse = np.fft.ifft2(image_fourier_shift)
    
    clean_image = abs(image_inverse)

    return clean_image


def clean_image_5(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'USA flag' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    A naive approach applied - picking the colors of the noise from the image, and filtering (by a median filter) the
    pixels containing those colors.

    Another approach is to calculate the likelihood function (with respect to a window average) and applying a threshold

    """

    r = 5

    # Specifying the colors of the noise (which we want to separate from the flag)
    colors = np.array([127, 32, 0, 136])
    clean_image = _clean_image_5_color(image, r, colors)
    
    # threshold = 0.001
    # std = 5
    # clean_image = _clean_img5_likelihood(image, r, std, threshold)

    return clean_image


def _clean_image_5_color(image: np.ndarray, r: int, colors: np.ndarray) -> np.ndarray:
    """
    A helper function

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
        r: The radius of the active receptive field of the filter
        colors: A numpy array containing the colors of which should a filter be applied upon

    Returns: A numpy array (NxD) representing the cleaned image

    """

    img = np.copy(image)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):

            # In case the pixel is a star's on the flag
            if row < 92 and col < 160:
                continue  # Skipping the stars on the flag

            # In case out of boundaries
            if col - r < 0 or col + r > img.shape[1]:
                continue
            
            mask = img[row, (col-r):(col + r+1)]  # A numpy nd array

            # If any of the colors matches the current pixel -> It belongs to a part of the noise
            if np.any(colors == img[row, col]):

                # Clean the pixel by applying a median filter
                img[row, col] = np.median(mask)
    
    return img


def _clean_image_5_likelihood(image: np.ndarray, r: int, std: int, threshold: float) -> np.ndarray:
    """
    A helper function

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
        r: The radius of the active receptive field of the filter
        threshold: The threshold of Likeliness

    Returns: A numpy array (NxD) representing the cleaned image

    """

    img = np.copy(image)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):

            # In case the pixel is a star's on the flag
            if row < 92 and col < 160:
                continue  # Skipping the stars on the flag

            # In case out of boundaries
            if col - r < 0 or col + r > img.shape[1]:
                continue
            
            mask = img[row, (col - r):(col + r+1)]  # A numpy array
            avg = np.average(mask)

            # The likelihood function (with respect to the window average)
            likelihood = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(img[row, col] - avg)**2 / (2 * std**2))

            if likelihood < threshold:  # Unlikely
                img[row, col] = np.median(mask)

    return img
            

def clean_image_6(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'cups' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    The 'cups' image needs a little enhancement of the low frequencies and decreasing of ringing.
    Hence, calculating the high and low pass of the image (creating an ideal high pass filter) and enhancing manually

    """
    
    image_fourier = np.fft.fft2(image)
    image_fourier_shift = np.fft.fftshift(image_fourier)
    
    center = np.array(image_fourier_shift.shape) // 2
    r = 20

    high_pass = np.copy(image_fourier_shift)

    # High pass ideal filter
    high_pass[(center[0] - r):(center[0] + r+1), (center[1] - r):(center[1] + r+1)] = 0

    # Creating a low pass image (in frequency domain)
    low_pass_image = image_fourier_shift - high_pass

    # Enhancing the low frequencies
    image_fourier_shift = high_pass + low_pass_image * 2
    
    clean_image = abs(np.fft.ifft2(image_fourier_shift))

    return clean_image


def clean_image_7(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'cups' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    The 'house' image suffers from a motion blur. Hence, trying to handle in frequency domain by creating a mask
    specified for motion blur

    """
    
    image_fourier = np.fft.fft2(image)

    # Creating a filter in frequency domain
    filter_image = np.zeros(image.shape)
    filter_image[0, :1] = 1
    filter_image[0, -9:] = 1
    filter_image = filter_image / filter_image.sum()

    # Back to image domain
    filter_fourier = np.fft.ifft2(filter_image)

    image_fourier_shift = np.fft.fftshift(image_fourier / filter_fourier)

    image_inverse = np.fft.ifft2(image_fourier_shift)

    clean_image = abs(image_inverse)

    return clean_image


def clean_image_8(image: np.ndarray) -> np.ndarray:
    """

    Args:
        image: A numpy array (NxD) representing the input image which needs to be cleaned
               This image is the 'bears' image in the 'Images' folder

    Returns: A numpy array (NxD) representing the cleaned image

    Method:
    The 'bears' image suffers from low contrast. we'll apply a contrast enhancement.

        """

    clean_image, _, _ = contrast_enhancement(image, [0, 255])

    return clean_image
