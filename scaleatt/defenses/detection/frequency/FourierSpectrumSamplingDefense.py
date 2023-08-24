import numpy as np
import typing

from defenses.detection.frequency.FrequencyDefense import FrequencyDefense
from scaling.ScalingApproach import ScalingApproach
from defenses.detection.frequency.spectrum.scoring.PeakSamplingStrategy import PeakSamplingStrategy


class FourierSpectrumSamplingDefense(FrequencyDefense):
    """
    -- Peak Spectrum from Paper --

    Defense based on analyzing Fourier spectrum, in particular,
    we analyze the peaks here properly.

    Wrapper for PeakSamplingStrategy class.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 sampling_radius: int,
                 peak_finding_method: typing.Callable):

        super().__init__(verbose, scaler_approach)
        self.peaksamplingstrategy: PeakSamplingStrategy = PeakSamplingStrategy(target_shape=scaler_approach.target_image_shape,
                                                                               peak_finding_method=peak_finding_method,
                                                                               sampling_radius=sampling_radius)

    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:
        score = self.peaksamplingstrategy.score(img=att_image)
        return score, {}
