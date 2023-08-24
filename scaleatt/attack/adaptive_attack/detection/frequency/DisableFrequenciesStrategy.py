import numpy as np
import typing

from defenses.detection.frequency.spectrum.utils import f_utils
from scaling.ScalingApproach import ScalingApproach

from .AdaptiveFourierAttackStrategy import AdaptiveFourierAttackStrategy





class DisableFrequenciesStrategy(AdaptiveFourierAttackStrategy):
    """
    Attack strategy where an already existing attack instance is given and some frequencies
    in spectrogram are disabled/reduced to prevent detection from Fourier detection mechanism
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 sampling_radius_attack: typing.Optional[int],
                 sampling_radius_defense: int,
                 peak_finding_method: typing.Callable = f_utils.get_main_peaks,
                 disabling_factor: float = 0.0):
        """
        Init adaptive attack against seak-spectrum detection by disabling/reducing frequency peaks
        :param verbose: show more information
        :param scaler_approach: scaler approach
        :param sampling_radius_attack: window size of reduced frequency values around each peak. If None,
        it is simply set to sampling_radius_defense.
        :param sampling_radius_defense: window size of defense for detection
        :param peak_finding_method: peak finding method
        :param disabling_factor: value to reduce frequencies, should be between 0 and 1
        """

        super().__init__(verbose=verbose, scaler_approach=scaler_approach)
        self.peak_finding_method = peak_finding_method

        if sampling_radius_attack is None:
            self.sampling_radius_attack = sampling_radius_defense
        else:
            self.sampling_radius_attack = int(sampling_radius_attack)

        self.disabling_factor: float = disabling_factor
        assert 0.0 <= self.disabling_factor <= 1.0, "disabling factor should lie in range [0,1]"

    # @Overwrite
    def _strategy(self, f_img: np.ndarray) -> np.ndarray:
        """
        Disables pixels in spectrograms around peaks (determined by some method) with a given radius

        :param f_img: spectrogram to manipulate
        :return: manipulated spectrogram
        """

        mask = f_utils.get_peaks_mask(f_img.shape,
                                      self.target_shape,
                                      self.peak_finding_method,
                                      self.sampling_radius_attack)

        f_img[mask] = f_img[mask] * self.disabling_factor

        return f_img

    def describe(self):
        print("AdaptiveAttackStrategy={}:\n"
              "\ttarget_shape={},\n"
              "\tpeak_finding_method={},\n"
              "\tsampling_radius_attack={},\n"
              "\tdisabling_factor={}".format(self.__class__.__name__,
                                            self.target_shape,
                                            self.peak_finding_method.__name__,
                                            self.sampling_radius_attack,
                                            self.disabling_factor))
