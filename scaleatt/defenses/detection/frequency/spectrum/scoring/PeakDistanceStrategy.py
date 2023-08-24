import numpy as np
from numpy.linalg import norm
import typing

from skimage.feature import peak_local_max

from defenses.detection.frequency.spectrum.utils import f_utils

from .ScoringStrategy import ScoringStrategy


class PeakDistanceStrategy(ScoringStrategy):
    """
    Wrapper for different scoring strategies where an image is given and a score is calculated based on some strategy
    """

    def __init__(self,
                 target_shape: typing.Tuple[int, int, int]):
        super().__init__(target_shape=target_shape)

    def strategy(self, f_img: np.ndarray) -> float:
        """
        Average distance of main peaks for peaks found by some peak finding algorithm in frequency space

        :param f_img: spectrogram for which to calculate score for
        :return: score
        """
        rows = f_img.shape[0]
        cols = f_img.shape[1]
        peaks = f_utils.get_main_peaks(f_img.shape, self.target_shape)

        distances = []
        for p in peaks:

            y_start = p[0] - self.target_shape[0] // 2
            y_end = p[0] + self.target_shape[0] // 2
            x_start = p[1] - self.target_shape[1] // 2
            x_end = p[1] + self.target_shape[1] // 2

            excerpt = f_img[max(y_start, 0): min(y_end, rows), max(x_start, 0): min(x_end, cols)]

            # it's a bit complicated to get the peak prediction right, as some excerpts may out of borders in parts
            local_prediction = np.array([self.target_shape[0] // 2, self.target_shape[1] / 2])
            if y_start < 0:
                local_prediction[0] = y_start + self.target_shape[0] // 2
            if x_start < 0:
                local_prediction[1] = x_start + self.target_shape[1] // 2

            coordinates = peak_local_max(excerpt, num_peaks=1)[0]  # is wrapped in extra dimension
            distance = norm(local_prediction - coordinates)
            distances.append(distance)

        return np.average(distances)

    def describe(self):
        print("ScoringStrategy={}:\n\ttarget_shape={}".format(self.__class__.__name__, self.target_shape))
