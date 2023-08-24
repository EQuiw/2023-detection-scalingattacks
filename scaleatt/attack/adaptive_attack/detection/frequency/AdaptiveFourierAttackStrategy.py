from abc import abstractmethod
import numpy as np
import typing

from defenses.detection.frequency.spectrum.utils import f_utils
from attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage2
from scaling.ScalingApproach import ScalingApproach



class AdaptiveFourierAttackStrategy(AdaptiveAttackOnAttackImage2):
    """
    Wrapper for different attack strategies where an already existing attack instance is given and some frequencies
    in spectrogram are altered to prevent detection from Fourier detection mechanism
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach):
        super().__init__(verbose, scaler_approach)

        self.target_shape = scaler_approach.target_image_shape
        assert len(self.target_shape) == 3 and self.target_shape[2] == 3

    def _get_manipulation(self, att: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply manipulation
        :return: manipulated output: R, G, B
        """

        f_r, f_g, f_b = f_utils.fourier_3_channels(att.copy())

        # replace_method = replace_at_peaks
        f_r = self._strategy(f_r)
        f_g = self._strategy(f_g)
        f_b = self._strategy(f_b)

        return f_r, f_g, f_b

    @abstractmethod
    def _strategy(self, f_img: np.ndarray) -> np.ndarray:
        """
        Concrete manipulation.

        :param f_img: spectrogram to manipulate
        :return: manipulated spectrogram
        """

    @abstractmethod
    def describe(self):
        pass

    # @Overwrite
    def counter_attack(self, att_image: np.ndarray, src: np.ndarray, tar: np.ndarray) -> typing.Tuple[np.ndarray, dict]:
        """
        Already crafted attack image with a known target_shape and apply adaptive manipulations in spectrogram.

        Splits channels, gets fshift by Fourier transform, disables pixels, does inverse transform, and glues channels
        back together.

        :param att_image: att to apply adaptive manipulations to
        :param src: source image (not used)
        :param tar: target image (not used)
        :return: adaptive_attack_image
        """
        att = att_image
        f_r, f_g, f_b = self._get_manipulation(att=att)

        # inverse r
        f_ishift_r = np.fft.ifftshift(f_r)
        img_r = np.real(np.fft.ifft2(f_ishift_r))

        # inverse g
        f_ishift_g = np.fft.ifftshift(f_g)
        img_g = np.real(np.fft.ifft2(f_ishift_g))

        # inverse b
        f_ishift_b = np.fft.ifftshift(f_b)
        img_b = np.real(np.fft.ifft2(f_ishift_b))

        # glue
        img_back = np.zeros(att.shape)
        for r in range(img_back.shape[0]):
            for c in range(img_back.shape[1]):
                img_back[r, c, 0] = img_r[r, c]
                img_back[r, c, 1] = img_g[r, c]
                img_back[r, c, 2] = img_b[r, c]

        img_back = self.scale_to_image_range(img_back, clip=False)
        img_back = img_back.astype("uint8")

        return img_back, {}

    @staticmethod
    def scale_to_image_range(img: np.ndarray, clip=False) -> np.ndarray:
        """Scale img to range 0-255"""
        assert img.dtype == np.float64, "I assume here an dtype of float64, others would work as well, but no uint8"
        if clip is True:  # assume image is in [0.0, 255.0] but some colors are out of bounds and need to be cut
            assert np.max(img) > 1.0
            img[0.0 > img] = 0.0
            img[255.0 < img] = 255.0
        new_img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
        return new_img