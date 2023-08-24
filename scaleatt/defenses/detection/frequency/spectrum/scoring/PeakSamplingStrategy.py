import numpy as np
import typing

from defenses.detection.frequency.spectrum.utils import f_utils

from .ScoringStrategy import ScoringStrategy


class PeakSamplingStrategy(ScoringStrategy):
    """
    Wrapper for different scoring strategies where an image is given and a score is calculated based on some strategy
    """

    def __init__(self,
                 target_shape: typing.Tuple[int, int, int],
                 peak_finding_method: typing.Callable = f_utils.get_main_peaks,
                 sampling_radius: int = 0):
        super().__init__(target_shape=target_shape)
        self.peak_finding_method = peak_finding_method
        self.sampling_radius = sampling_radius

    def strategy(self, f_img: np.ndarray) -> float:
        """
        Percentile of score of average intensity at supposed peaks in fourier spectrogram sampled with radius

        :param f_img: spectrogram for which to calculate score for
        :return: score
        """

        mask = f_utils.get_peaks_mask(f_img.shape,
                                      self.target_shape,
                                      self.peak_finding_method,
                                      self.sampling_radius)
        score = np.average(f_img[mask])

        # we could also try the relation to the overall average,
        # but taking the percentile of score got better roc-curves
        # score = score / np.average(img)
        score = ScoringStrategy.get_percentile_of_score(f_img, score)

        return score

    def describe(self):
        print("ScoringStrategy={}:\n"
              "\ttarget_shape={},\n"
              "\tpeak_finding_method={},\n"
              "\tsampling_radius={}".format(self.__class__.__name__,
                                            self.target_shape,
                                            self.peak_finding_method.__name__,
                                            self.sampling_radius))
