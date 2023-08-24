from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from scaling.ScalingApproach import ScalingApproach


class AdaptiveAttack(ABC):
    """
    Adaptive attack against a defense.
    Note that we have two abstract subclasses. A defense might adapt the
    attack image(AdaptiveAttackOnAttackImage), or create a new one that misleads a defense (AdaptiveAttackOnSrcTarImage)
    """

    def __init__(self, verbose: bool, scaler_approach: ScalingApproach):
        self.scaler_approach = scaler_approach
        self.verbose = verbose


class AdaptiveAttackOnAttackImage(AdaptiveAttack):

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 ):
        super().__init__(verbose, scaler_approach)


    @abstractmethod
    def counter_attack(self, att_image: np.ndarray) -> np.ndarray:
        """
        Counter attack: adapt attack image
        :param att_image:
        :return: a changed attack image to mislead defense.
        """
        pass

    @abstractmethod
    def get_stats_last_run(self):
        """
        Return stats from last run(s).
        """


class AdaptiveAttackOnSrcTarImage(AdaptiveAttack):

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 ):
        super().__init__(verbose, scaler_approach)


    @abstractmethod
    def counter_attack(self, src_image: np.ndarray, tar_image: np.ndarray) -> np.ndarray:
        """
        Counter attack: create new attack image
        :param src_image:
        :param tar_image:
        :return: a novel attack image to mislead defense.
        """
        pass


class AdaptiveAttackOnAttackImage2(AdaptiveAttack):
    """
    Newer API for AdaptiveAttackOnAttackImage
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 ):
        super().__init__(verbose, scaler_approach)

    @abstractmethod
    def counter_attack(self, att_image: np.ndarray, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Counter attack: adapt attack image
        :param att_image: attack image
        :param src: source image
        :param tar: target image (added to make it more easily extendable in future)
        :return: a changed attack image to mislead defense + maybe more infos such as stats from current call
        """
        pass