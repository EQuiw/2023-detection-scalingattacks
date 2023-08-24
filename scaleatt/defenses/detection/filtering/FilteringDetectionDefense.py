import numpy as np
import typing
import cv2

from defenses.detection.DetectionTypeDefense import DetectionTypeDefense
from defenses.detection.DetectionDefense import DetectionDefense
from scaling.ScalingApproach import ScalingApproach
from utils.SimilarityMeasurementTool import SimilarityMeasurementTool
from utils.SimilarityMeasure import SimilarityMeasure


class FilteringDetectionDefense(DetectionDefense):
    """
    -- Maximum Filter and Minimum Filter from Paper --

    Detection defense based on applying min or max-filter on input image, and then comparing filtered image
    and input image --- as suggested by Kim et al.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 detection_method: DetectionTypeDefense,
                 similarity_measurement: SimilarityMeasure,
                 kernel_size: typing.Tuple[int, int],
                 ):
        """
        Init detection method.
        :param verbose: show more information
        :param scaler_approach: scaling settings
        :param detection_method: specify type of detection method {min filter, max filter}. Other detection types
        are not supported here!
        :param similarity_measurement: specify type of metric
        :param kernel_size: kernel size of filter
        """
        super().__init__(verbose, scaler_approach)
        self.detection_method: DetectionTypeDefense = detection_method
        self.similarity_measurement: SimilarityMeasure = similarity_measurement
        self.kernel_size = kernel_size

    # @Overwrite
    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:

        if self.detection_method == DetectionTypeDefense.filtering_min:
            filtered_image = self.apply_min_filter(image_of_interest=att_image, kernel_size=self.kernel_size)
        elif self.detection_method == DetectionTypeDefense.filtering_max:
            filtered_image = self.apply_max_filter(image_of_interest=att_image, kernel_size=self.kernel_size)
        else:
            raise NotImplementedError()

        score = SimilarityMeasurementTool.sim_measure(img1=att_image, img2=filtered_image,
                                                      similarity_measurement=self.similarity_measurement)
        if self.verbose:
            print("{}, {} got scores: {}".format(self.detection_method.name, self.similarity_measurement.name, score))

        return score, {} # {'filtered_image': filtered_image}

    @staticmethod
    def apply_min_filter(image_of_interest: np.ndarray,
                         kernel_size: typing.Tuple[int, int] = (3, 3)) -> np.ndarray:
        """
        Applies min filter on image.
        :param image_of_interest: input image
        :param kernel_size: kernel size for min-filtering
        :return: min-filtered image
        """
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, kernel_size)
        min_image = cv2.erode(image_of_interest, kernel)
        return min_image

    @staticmethod
    def apply_max_filter(image_of_interest: np.ndarray,
                         kernel_size: typing.Tuple[int, int] = (3, 3)) -> np.ndarray:
        """
        Applies max filter on image.
        :param image_of_interest: input image
        :param kernel_size: kernel size for max-filtering
        :return: max-filtered image
        """
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, kernel_size)
        max_image = cv2.dilate(image_of_interest, kernel)
        return max_image

