import numpy as np

from defenses.detection.frequency.spectrum.utils import f_utils
from scaling.ScalingApproach import ScalingApproach

from .AdaptiveFourierAttackStrategy import AdaptiveFourierAttackStrategy


class AddFrequencyPeakStrategy(AdaptiveFourierAttackStrategy):
    """
    Attack strategy where an already existing attack instance is given and some frequencies
    in spectrogram are added far away from main peaks in each excerpt to prevent detection
    from Fourier detection mechanism.

    Excerpts are tiles in target_shape, starting from the center
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   x   .   .   .   .   .
    .   .   ---------   .   .   .   .   .   .
    .   .   |   .   |   .   .   .   .   .   .
    .   .   ---------   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .   .   .   .

    Note that we use the symmetry of the fourier transform to add peaks. We add peaks in the corners + 5 in each
    dimension. This causes peaks in the other corner in excerpts on the opposite. See the spectrum as example.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 frequency_modification_factor: int = 0):
        super().__init__(verbose=verbose, scaler_approach=scaler_approach)
        self.frequency_modification_factor = frequency_modification_factor

    # @Overwrite
    def _strategy(self, f_img: np.ndarray) -> np.ndarray:
        """
        Add a bright pixel to distract peak finding algorithm from center of excerpt

        :param f_img: spectrogram to manipulate
        :return: manipulated spectrogram
        """
        peaks = f_utils.get_main_peaks(f_img.shape, self.target_shape)

        center = (int(f_img.shape[0] / 2), int(f_img.shape[1] / 2))

        for p in peaks:

            y_start = p[0] - self.target_shape[0] // 2
            y_end = p[0] + self.target_shape[0] // 2
            x_start = p[1] - self.target_shape[1] // 2
            x_end = p[1] + self.target_shape[1] // 2

            fake_peak_coordinates = np.array([y_start, x_start])  # default: just increase the intensity at upper left pixel
            if y_start < center[0]:  # correct if excerpt might be cropped in axis 0
                fake_peak_coordinates[0] = y_end
            if x_start < center[1]:  # correct if excerpt might be cropped in axis 1
                fake_peak_coordinates[1] = x_end

            f_img[fake_peak_coordinates[0] + 5,
                  fake_peak_coordinates[1] + 5] = np.max(f_img) / self.frequency_modification_factor

        return f_img

    def describe(self):
        print("AdaptiveAttackStrategy={}:\n"
              "\ttarget_shape={},\n"
              "\tpeak_finding_method={}".format(self.__class__.__name__,
                                                self.target_shape,
                                                self.frequency_modification_factor))
