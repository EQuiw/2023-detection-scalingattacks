import numpy as np
from abc import abstractmethod
import typing

from defenses.detection.DetectionDefense import DetectionDefense
from scaling.ScalingApproach import ScalingApproach


class FrequencyDefense(DetectionDefense):
    """
    Generic upper class for
    detection defenses based on analyzing frequency spectrum.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach):

        super().__init__(verbose, scaler_approach)

    @abstractmethod
    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:
        pass

