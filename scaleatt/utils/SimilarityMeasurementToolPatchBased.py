# This file contains all methods to compare two images by dividing them into meaningful blocks/patches
# and then compare these blocks individually. This approach is especially suited to detect backdoors.
import numpy as np
import cv2 as cv
from enum import Enum
from skimage.util.shape import view_as_windows
from typing import Tuple

from utils.SimilarityMeasurementTool import SimilarityMeasurementTool
from utils.SimilarityMeasure import SimilarityMeasure


class PatchMethod(Enum):
    """
    Choice what patch method shall be used
    """
    use_block_based = 124
    use_segmentation_based = 256


def sim_measure_patchwise(patch_method: PatchMethod,
                          img1: np.ndarray,
                          img2: np.ndarray,
                          similarity_measurement: SimilarityMeasure,
                          **kwargs) -> np.ndarray:
    """
    If we want to detect a backdoor with a down-and-upscaling-based defense,
    img1 should be the down-and-upscaled image! See @_compare_images_segmentationbased why.
    :param patch_method: Method to divide image.
    :param img1: image to be compared.
    :param img2: image to be compared
    :param similarity_measurement: similarity measure, such as PSNR
    :param kwargs: parameters for patch method
    :return: SimilarityMeasure scores from each patch.
    """

    if patch_method == PatchMethod.use_block_based:
        return _compare_images_blockbased(img1=img1,
                                          img2=img2,
                                          similarity_measurement=similarity_measurement,
                                          **kwargs)
    elif patch_method == PatchMethod.use_segmentation_based:
        return _compare_images_segmentationbased(img1=img1,
                                                 img2=img2,
                                                 similarity_measurement=similarity_measurement,
                                                 **kwargs)
    else:
        raise NotImplementedError("sim_measure_patchwise, passed patch_method does not exist")


def compute_block_stride_size_from_factors(image_shape: Tuple,
                                           window_factor: int,
                                           step_size_factor: int):
    assert len(image_shape) == 2 or len(image_shape) == 3  # we ignore the channel dimension
    window_shape = (image_shape[0] // window_factor, image_shape[1] // window_factor)
    step_size = max(window_shape[0] // step_size_factor, 1)

    return window_shape, step_size


def _compare_images_blockbased(img1: np.ndarray,
                               img2: np.ndarray,
                               similarity_measurement: SimilarityMeasure,
                               window_factor: int = 10,
                               step_size_factor: int = 2) -> np.ndarray:
    """
    Divide image into sub-blocks and compare these blocks using passed similarity_measurement.
    This method will return an array with the scores from each sub-block.

    The parameters window_factor and step_size_factor are used to compute the block sizes and strides.
    """

    window_shape, step_size = compute_block_stride_size_from_factors(
        image_shape=img1.shape, window_factor=window_factor, step_size_factor=step_size_factor
    )
    img1 = img1.copy()
    img2 = img2.copy()

    # 1. Split into windows
    if len(img1.shape)==2:
        img1 = img1[:, :, np.newaxis]
        img2 = img2[:, :, np.newaxis]
    elif len(img1.shape) == 3:
        assert img1.shape[2] == 3
    else: raise NotImplementedError("Img Shape, too many dimensions")

    windows_img1 = []
    windows_img2 = []

    pad_top, pad_bottom, pad_left, pad_right = _get_padding(in_height=img1.shape[0], in_width=img1.shape[1],
                                                            filter_height=window_shape[0], filter_width=window_shape[1],
                                                            stride_height=step_size, stride_width=step_size)

    for i in range(img1.shape[2]):
        img1_single = img1[:, :, i]
        img2_single = img2[:, :, i]

        # a. Padding
        img1_pad = np.pad(img1_single, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')
        img2_pad = np.pad(img2_single, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')

        # b. Get blocks
        img1_pad_windows = view_as_windows(img1_pad, window_shape, step=step_size)
        img2_pad_windows = view_as_windows(img2_pad, window_shape, step=step_size)
        windows_img1.append(img1_pad_windows)
        windows_img2.append(img2_pad_windows)

    comparison_values = []
    assert len(windows_img1[0].shape) == 4
    r1 = windows_img1[0]
    r2 = windows_img2[0]
    for i in range(r1.shape[0]):
        for j in range(r1.shape[1]):

            if len(windows_img1) == 1:
                current_block1 = r1[i, j, :, :]
                current_block2 = r2[i, j, :, :]
            else:
                current_block1 = np.stack((r1[i, j, :, :], windows_img1[1][i, j, :, :], windows_img1[2][i, j, :, :]),axis=2)
                current_block2 = np.stack((r2[i, j, :, :], windows_img2[1][i, j, :, :], windows_img2[2][i, j, :, :]),axis=2)

            if np.sum(current_block1-current_block2) == 0:
                continue # avoid inf in PSNR

            score = SimilarityMeasurementTool.sim_measure(img1=current_block1, img2=current_block2,
                                                          similarity_measurement=similarity_measurement)
            if score < np.inf and not np.isnan(score):
                comparison_values.append(score)

    return np.array(comparison_values)


def compare_images_blockbased_targeted(img1: np.ndarray,
                                             img2: np.ndarray,
                                             binary_mask: np.ndarray,
                                             quantile_value: float,
                                             window_factor: int = 10,
                                             step_size_factor: int = 2) -> np.ndarray:
    """
    Divide image into sub-blocks and compare these blocks using passed similarity_measurement.
    This method will return an array with the scores from each sub-block.
    Diff to above: we operate only on values defined by binary mark and use the quantile value
    to determine the similarity in each block.
    todo should be merged with fct. above in future.

    The parameters window_factor and step_size_factor are used to compute the block sizes and strides.
    """

    window_shape, step_size = compute_block_stride_size_from_factors(
        image_shape=img1.shape, window_factor=window_factor, step_size_factor=step_size_factor
    )
    img1 = img1.copy()
    img2 = img2.copy()

    # 1. Split into windows
    if len(img1.shape)==2:
        img1 = img1[:, :, np.newaxis]
        img2 = img2[:, :, np.newaxis]
    elif len(img1.shape) == 3:
        assert img1.shape[2] == 3
    else: raise NotImplementedError("Img Shape, too many dimensions")

    windows_img1 = []
    windows_img2 = []

    pad_top, pad_bottom, pad_left, pad_right = _get_padding(in_height=img1.shape[0], in_width=img1.shape[1],
                                                            filter_height=window_shape[0], filter_width=window_shape[1],
                                                            stride_height=step_size, stride_width=step_size)

    binary_mask_pad = np.pad(binary_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')
    binary_mask_pad_windows = view_as_windows(binary_mask_pad, window_shape, step=step_size)

    for i in range(img1.shape[2]):
        img1_single = img1[:, :, i]
        img2_single = img2[:, :, i]

        # a. Padding
        img1_pad = np.pad(img1_single, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')
        img2_pad = np.pad(img2_single, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')

        # b. Get blocks
        img1_pad_windows = view_as_windows(img1_pad, window_shape, step=step_size)
        img2_pad_windows = view_as_windows(img2_pad, window_shape, step=step_size)
        windows_img1.append(img1_pad_windows)
        windows_img2.append(img2_pad_windows)

    comparison_values = []
    assert len(windows_img1[0].shape) == 4
    r1 = windows_img1[0]
    r2 = windows_img2[0]
    for i in range(r1.shape[0]):
        for j in range(r1.shape[1]):

            if len(windows_img1) == 1:
                current_block1 = r1[i, j, :, :]
                current_block2 = r2[i, j, :, :]
            else:
                current_block1 = np.stack((r1[i, j, :, :], windows_img1[1][i, j, :, :], windows_img1[2][i, j, :, :]),axis=2)
                current_block2 = np.stack((r2[i, j, :, :], windows_img2[1][i, j, :, :], windows_img2[2][i, j, :, :]),axis=2)

            current_block1_filtered = current_block1[binary_mask_pad_windows[i, j, :, :].astype(bool)]
            current_block2_filtered = current_block2[binary_mask_pad_windows[i, j, :, :].astype(bool)]

            val1 = np.abs(current_block1_filtered.reshape(-1).astype(np.int) - current_block2_filtered.reshape(-1).astype(np.int))
            score = np.quantile(val1, quantile_value)

            # if np.sum(current_block1-current_block2) == 0:
            #     continue # avoid inf in PSNR

            if score < np.inf and not np.isnan(score):  # should not happen, but for consistency here.
                comparison_values.append(score)

    return np.array(comparison_values)


def _get_padding(in_height: int,
                 in_width: int,
                 stride_height: int,
                 stride_width: int,
                 filter_height: int,
                 filter_width: int):
    """
    Interntal helper function for window_based comparison.
    """

    if in_height % stride_height == 0:
        pad_along_height = max(filter_height - stride_height, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_height), 0)
    if in_width % stride_width == 0:
        pad_along_width = max(filter_width - stride_width, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_width), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_top, pad_bottom, pad_left, pad_right


# from utils.plot_image_utils import plot_images1
def _compare_images_segmentationbased(img1: np.ndarray, img2: np.ndarray, k_noclusters: int,
                                      similarity_measurement: SimilarityMeasure = SimilarityMeasure.PSNR,
                                      no_threads: int = 1) -> np.ndarray:
    """
    Divide image into segments and compare these segments using passed similarity_measurement.
    This method will return an array with the scores from each sub-block.

    Segmentation is computed w.r.t to img1. Then, img1 and img2 are compared on segments.

    Thus, if we want to detect a backdoor, img1 should be the image where the backdoor is clearly visible.
    For instance, if we down- and upscale an image, img1 should be the down-and-upscaled image!
    """
    cv.setNumThreads(no_threads)

    img1 = img1.copy()

    assert len(img1.shape)==3 or len(img1.shape==2)
    if len(img1.shape) == 3:
        assert img1.shape[2] == 3
        twodimage = img1.reshape((-1,3))
    else:
        twodimage = img1.reshape(-1)
        twodimage = np.expand_dims(twodimage, axis=1)

    twodimage = np.float32(twodimage)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts=10
    ret,label,center=cv.kmeans(twodimage,k_noclusters,None,criteria,attempts,cv.KMEANS_PP_CENTERS)

    comparison_values = []
    for k_cluster in range(1, k_noclusters):
        masked_image = np.zeros(img1.shape)
        labels_matrix = label.reshape((img1.shape[0], img1.shape[1]))
        masked_image[labels_matrix == k_cluster] = 1
        # plot_images1(masked_image)

        current_block1=img1[masked_image==1]
        current_block2=img2[masked_image==1]
        score = SimilarityMeasurementTool.sim_measure(img1=current_block1, img2=current_block2,
                                                      similarity_measurement=similarity_measurement)
        if score < np.inf and not np.isnan(score):
            comparison_values.append(score)

    return np.array(comparison_values)


