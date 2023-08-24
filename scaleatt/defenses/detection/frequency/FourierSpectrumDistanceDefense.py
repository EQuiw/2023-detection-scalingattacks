import numpy as np
import typing

from defenses.detection.frequency.FrequencyDefense import FrequencyDefense
from scaling.ScalingApproach import ScalingApproach
from defenses.detection.frequency.spectrum.scoring.PeakDistanceStrategy import PeakDistanceStrategy


class FourierSpectrumDistanceDefense(FrequencyDefense):
    """
    -- Peak Distance from Paper --

    Defense based on analyzing Fourier spectrum, in particular,
    we analyze the peaks here properly.

    Wrapper for PeakDistanceStrategy class.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach):

        super().__init__(verbose, scaler_approach)
        self.peakdistancestrategy: PeakDistanceStrategy = PeakDistanceStrategy(target_shape=scaler_approach.target_image_shape)

    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:
        score = self.peakdistancestrategy.score(img=att_image)
        return score, {}
