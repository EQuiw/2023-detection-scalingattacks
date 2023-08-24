from abc import ABC, abstractmethod
import numpy as np
import typing

from scipy.stats import percentileofscore

from defenses.detection.frequency.spectrum.utils import f_utils


class ScoringStrategy(ABC):
    """
    Wrapper for different scoring strategies where an image is given and a score is calculated based on some strategy
    """

    def __init__(self,
                 target_shape: typing.Tuple[int, int, int]):
        self.target_shape = target_shape

    @abstractmethod
    def strategy(self, f_img: np.ndarray) -> float:
        pass

    @abstractmethod
    def describe(self):
        pass

    def score(self, img: np.ndarray) -> float:
        """
        Take image, get spectrogram, calc score by strategy

        :param img: image to calculate score for
        :return: score
        """
        f_img = f_utils.fourier(img)
        return self.strategy(f_img)

    @staticmethod
    def get_percentile_of_score(img: np.ndarray,
                                score: float) -> float:
        """
        Percentile of score for flattened image

        :param img: image
        :param score: score
        :return: percentile of score
        """
        return percentileofscore(img.flatten(), score)
