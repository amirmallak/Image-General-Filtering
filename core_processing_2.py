import numpy as np
import matplotlib.pyplot as plt

from typing import List
from cv2 import cv2


def write_morphing_video(image_list: List[np.ndarray], video_name: str):
    out = cv2.VideoWriter(video_name+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20.0, image_list[0].shape, 0)
    for image in image_list:
        out.write(image)
    out.release()


def create_morph_sequence(image_1: np.ndarray, image_1_pts: np.ndarray, image_2: np.ndarray, image_2_pts: np.ndarray,
                          t_list: List[float], transform_type: int) -> List[np.ndarray]:
    """
    Input:
    image_1, image_2 -- grayscale image arrays in the range [0..255]
    image_1_pts, image_2_pts -- np arrays Nx3 of chosen coordinates of image_1, image_2
    t_list –- a vector with t values where t is in [0..1]
              We can use np.linspace(0,1,M) (outside the function) to create a vector with M equal steps between 0 and 1
    transform_type -– a scalar. 0 = affine, 1 = projective.

    Output:
    images_list –- a list of images. The same size as t_list.

    Method:
    Input images must be of same size. We calculate the transforms transform_12 and transform_21 which maps
    image_1 to image_2 and image_2 to image_1 respectively.
    For every t we do the following: We calculate transform_12_t, the transformation from image_1 to image_2,
    t parts of the way. As we did in the Algorithm:
    𝑇_12_𝑡 = (1−𝑡) * (1 0 0 0 1 0 0 0 1) + 𝑡 * 𝑇_12 .
    We Calculate T21_1_t, the transformation from image_2 to image_1, (1-t) parts of the way:
    𝑇_21_𝑡 = (1−𝑡) * 𝑇_21 + 𝑡 * (1 0 0 0 1 0 0 0 1).
    We map image_1 using T12_t, producing new_image_1. And map image_2 using T21_1_t, producing new_image_2
    We crossdisolve new_image_1 and new_image_2 with weights associated with t.

    Note:
     We use function np.eye(3) to create I matrix of size 3x3.
     We use functions inside create_morph_sequence:
        o find_projective_transform/find_affine_transform to acquire transform.
        o map_image to map to new_image_1, new_image_2.
    """

    if transform_type:
        # Projective transforms
        transform_12: np.ndarray = find_projective_transform(image_2_pts, image_1_pts)
        transform_21: np.ndarray = find_projective_transform(image_1_pts, image_2_pts)
    else:
        # Affine transforms
        transform_12: np.ndarray = find_affine_transform(image_2_pts, image_1_pts)
        transform_21: np.ndarray = find_affine_transform(image_1_pts, image_2_pts)

    images_list = []

    for t in t_list:
        # Calculate new_image for each t
        transform_12_t: np.ndarray = (1 - t) * np.eye(3) + t * transform_12
        transform_21_t: np.ndarray = (1 - t) * transform_21 + t * np.eye(3)
        map_image_1: np.ndarray = map_image(image_1, transform_12_t, image_2.shape)
        map_image_2: np.ndarray = map_image(image_2, transform_21_t, image_1.shape)
        new_image = np.round(map_image_1 * (1 - t) + map_image_2 * t)
        images_list.append(new_image.astype('uint8'))

    return images_list


def map_image(image: np.ndarray, transformation: np.ndarray, size_out_image: tuple) -> np.ndarray:
    """
    Input:
    image -- a grayscale image array in the range [0..255]
    transformation -- a 3x3 matrix representing a transformation
    size_out_image –- a tuple (numRows, numCols) representing the size of the output image

    Return:
    new_image -- a grayscale image in the range [0..255] of size size_out_image containing the transformed image

    Method:
    Create new_image as an empty array of size size_out_image.
    For every coordinate of new_image, applying inverse mapping to determine source coordinates.
    We Map value of image at the source coordinates to the new image.
    then Using a bilinear Interpolation (tip: One could implement nearest neighbor interpolation first as it is easier
    to code, and once everything is running – one can replace it with bilinear interpolation).
    We should ensure that source coordinates do not fall outside image range.
    (points whose source are outside source image are assigned gray value zero).

    Note:
     We use functions np.linalg.inv , np.meshgrid, np.vstack, np.matmul, np.round, np.any, np.delete.
     We do not iterate over pixels of image.
     We don’t need a for loop to implement this function.
    """

    new_image: np.ndarray = np.zeros(size_out_image)  # Shape: (R x C) of target image
    dest_row_count = size_out_image[0]
    dest_col_count = size_out_image[1]

    # create meshgrid of all coordinates in new image [x,y]
    coord_mesh = np.meshgrid(np.arange(dest_row_count), np.arange(dest_col_count))
    coord_mesh = np.vstack([coord_mesh[0].flatten(), coord_mesh[1].flatten()])

    # add homogenous coord [x,y,1]
    coord_mesh_h = np.vstack([coord_mesh, np.ones(coord_mesh.shape[1])])

    # calculate source coordinates that correspond to [x,y,1] in new image
    source_coord = transformation @ coord_mesh_h
    source_coord = source_coord[:2, :] / source_coord[2, :]  # Normalizing as for the Projection transformation

    # find coordinates outside range and delete (in source and target)

    del_indices = set()
    del_indices.update(np.where(source_coord[0, :] < 0)[0])
    del_indices.update(np.where(source_coord[1, :] < 0)[0])
    del_indices.update(np.where(source_coord[0, :] > image.shape[0])[0])
    del_indices.update(np.where(source_coord[1, :] > image.shape[1])[0])
    source_coord_del = np.delete(source_coord, list(del_indices), axis=1)
    coord_mesh_del = np.delete(coord_mesh, list(del_indices), axis=1)

    # interpolate - bilinear
    source_coord_del[0, np.where(source_coord_del[0, :] > (image.shape[0] - 1))] -= 1
    source_coord_del[1, np.where(source_coord_del[1, :] > (image.shape[1] - 1))] -= 1

    source_coord_del_ceil = np.ceil(source_coord_del).astype('int')
    source_coord_del_floor = np.floor(source_coord_del).astype('int')

    epsilon = 1e-10
    source_coord_del_weights = (source_coord_del - source_coord_del_floor) / \
                               (source_coord_del_ceil - source_coord_del_floor + epsilon)
    # source_coord_del_weights = np.nan_to_num(source_coord_del_weights, nan=1e-2)

    im_bilinear_row_floor = image[source_coord_del_floor[0], source_coord_del_floor[1]] * (1 - source_coord_del_weights[1]) + \
                            image[source_coord_del_floor[0], source_coord_del_ceil[1]] * source_coord_del_weights[1]
    im_bilinear_row_ceil = image[source_coord_del_ceil[0], source_coord_del_floor[1]] * (1 - source_coord_del_weights[1]) + \
                           image[source_coord_del_ceil[0], source_coord_del_ceil[1]] * source_coord_del_weights[1]
    im_bilinear = im_bilinear_row_floor * (1 - source_coord_del_weights[0]) + \
                  im_bilinear_row_ceil * source_coord_del_weights[0]

    # apply corresponding coordinates
    new_image[coord_mesh_del[0], coord_mesh_del[1]] = im_bilinear

    return new_image


def find_projective_transform(points_set_1: np.ndarray, points_set_2: np.ndarray) -> np.ndarray:
    """
    Input:
    points_set_1, points_set_2 -- arrays Nx3 of coordinates. Representing corresponding points between the 2 sets

    Return:
    t_projective –- a 3x3 np matrix representing a projective transformation

    Method:
    Calculates the parameters of the projective transform that best maps points in points_set_1 to corresponding points
    in points_set_2 in the least mean square sense. ℎ_𝑝𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑣𝑒 = (𝑎 𝑏 𝑒 𝑐 𝑑 𝑓 𝑔 ℎ 1) - a 3x3 matrix.
    As for Geometric Operations

    Note:
     We could loop over the N points.
    """

    number_of_points: int = points_set_1.shape[0]
    # px - Source Image x Coordinates
    px: np.ndarray = points_set_1[:, 0]  # Shape: (N x 1)
    # py - Source Image y Coordinates
    py: np.ndarray = points_set_1[:, 1]  # Shape: (N x 1)
    # ptx - Target Image x Coordinates
    ptx: np.ndarray = points_set_2[:, 0]  # Shape: (N x 1)
    # pty - Target Image y Coordinates
    pty: np.ndarray = points_set_2[:, 1]  # Shape: (N x 1)

    x_matrix = np.zeros([number_of_points * 2, 8])  # Shape: (2N x 8)
    xt_matrix = np.zeros(number_of_points * 2)  # Shape: (N x 2)

    for i in range(0, number_of_points):
        x_matrix[i*2, :] = np.array([px[i], py[i], 0, 0, 1, 0, -px[i] * ptx[i], -py[i] * ptx[i]])
        x_matrix[i*2 + 1, :] = np.array([0, 0, px[i], py[i], 0, 1, -px[i] * pty[i], -py[i] * pty[i]])
        xt_matrix[i*2] = points_set_2[i, 0]
        xt_matrix[i*2 + 1] = points_set_2[i, 1]

    # Calculating the Projective Matrix Transformation
    a, b, c, d, e, f, g, h = np.linalg.pinv(x_matrix) @ xt_matrix
    # t_projective - Projective Matrix Transformation
    t_projective = np.array([[a, b, e], [c, d, f], [g, h, 1]])  # Shape: (3 x 3)

    return t_projective


def find_affine_transform(points_set_1: np.ndarray, points_set_2: np.ndarray) -> np.ndarray:
    """
    Input:
    points_set_1, points_set_1 -- arrays Nx3 of coordinates representing corresponding points between the 2 sets

    Return:
    t_affine -– a 3x3 np matrix representing an affine transformation

    Method:
    Calculates the parameters of the affine transform that best maps points in points_set_1 to corresponding points in
    points_set_2 in the least mean square sense. ℎ_𝑎𝑓𝑓𝑖𝑛𝑒 = (𝑎 𝑏 𝑒 𝑐 𝑑 𝑓 0 0 1) - a 3x3 matrix.

    Note:
     We use functions np.matmul, np.linalg.pinv, and np.reshape. As for Geometric Operations.
     We could loop over the N points.
    """

    number_of_points: int = points_set_1.shape[0]  # Recall number_of_points = N
    # px - Source Image x Coordinates
    px: np.ndarray = points_set_1[:, 0]  # Shape: (N x 1)
    # py - Source Image y Coordinates
    py: np.ndarray = points_set_1[:, 1]  # Shape: (N x 1)

    x_matrix = np.zeros([number_of_points * 2, 6])  # Shape: (2N x 6)
    xt_matrix = np.zeros(number_of_points * 2)  # Shape: (2N)

    for i in range(0, number_of_points):
        x_matrix[i*2, :] = np.array([px[i], py[i], 0, 0, 1, 0])
        x_matrix[i*2 + 1, :] = np.array([0, 0, px[i], py[i], 0, 1])
        xt_matrix[i*2] = points_set_2[i, 0]
        xt_matrix[i*2 + 1] = points_set_2[i, 1]

    # Calculating the Affine Matrix Transformation
    a, b, c, d, e, f = np.linalg.pinv(x_matrix) @ xt_matrix
    # t_affine - Affine Matrix Transformation
    t_affine = np.array([[a, b, e], [c, d, f], [0, 0, 1]])  # Shape: (3 x 3)

    return t_affine


def image_resizing(image_1: np.ndarray, image_2: np.ndarray) -> (np.ndarray, np.ndarray):
    # Resizing the bigger image to the size of the smaller one
    bigger_image = image_1 if (image_1.size > image_2.size) else image_2
    smaller_image = image_1 if (image_1.size <= image_2.size) else image_2
    smaller_image_dim = (smaller_image.shape[1], smaller_image.shape[0])
    resized_bigger_image = cv2.resize(bigger_image, smaller_image_dim, interpolation=cv2.INTER_CUBIC)

    if bigger_image is image_1:
        return resized_bigger_image, smaller_image

    return smaller_image, resized_bigger_image


def _get_image_points(image, var_name: str, number_of_points: int) -> None:
    plt.figure()
    plt.imshow(image)
    plt.title(f'Choose {number_of_points} points from the below image that you wish to transform')
    # Recall number_of_points = N
    image_points: List[tuple] = plt.ginput(n=number_of_points, timeout=0)  # Shape: (1 x 2N)
    plt.close()
    image_points: np.ndarray = np.array(image_points)  # Shape: (N x 2)
    # Reversing the coordinates (for in plt.ginput() function - x is the horizontal axis, and y is the vertical)
    image_points = image_points[:, ::-1]  # Now the x coordinate in image_points fits an x coordinate in an image
    image_points = np.round(image_points)
    ones_array = np.ones(number_of_points)  # Shape: (1 x N)
    # Adding a new dimension to each coordinate
    image_points = np.column_stack([image_points, ones_array])  # Shape: (N x 3)

    np.save(var_name + ".npy", image_points)


def get_image_points(image_1, image_2, var_name_1: str, var_name_2: str, number_of_points: int = 12) -> None:
    """
    Input:
    image_1, image_2 -- grayscale images in the range [0..255]. Not necessarily same size
    var_name_1, var_name_2 -– strings that represent the names of variables to be saved as
                              (for example if we want to save the number_of_points as imagePts1.npy and imagePts2.npy,
                              we will pass the names var_name_1 = “imagePts1” and var_name_2 = “imagePts2”
    number_of_points -– number of points the user chooses

    Return:
    None

    Method:
    Allows user to select corresponding pairs of points, one set in image_1 and one in image_2 (in same order).
    Function opens a figure of image_1 and let’s the user select number_of_points on the image.
    Then does the same for image_2.
     Coordinates (x,y) of the points in each image are collected in imagePts1, imagePts2 – np arrays Nx2.
     After collecting all coordinates, we round them using np.round and add a third dimension of ones which will
       make them Nx3 arrays, then save the array using: np.save(“imagePts1.npy”, imagePts1),
       np.save(“imagePts2.npy”, imagePts2). This will save the arrays as files in the project folder. So later, we can
       load them instead of choosing points again, by using:
       imagePts1 = np.load(“imagePts1.npy”) imagePts2 = np.load(“imagePts2.npy”)
     We use plt.ginput to interactively choose points.
       Documentation: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.ginput.html

    Note:
     Pay very close attention to the x,y order. In Image assignment image[x,y] – x is the rows, but in
       x,y = ginput – x is the horizontal axis.
    """

    # resized_bigger_image, smaller_image = image_resizing(image_1, image_2)

    _get_image_points(image_1, var_name_1, number_of_points)
    _get_image_points(image_2, var_name_2, number_of_points)
