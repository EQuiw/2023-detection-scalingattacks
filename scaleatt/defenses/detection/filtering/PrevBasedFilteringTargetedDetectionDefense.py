import numpy as np
import typing

from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector
from defenses.detection.DetectionDefense import DetectionDefense
from scaling.ScalingApproach import ScalingApproach
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefense import PreventionDefense


class PrevBasedFilteringTargetedDetectionDefense(DetectionDefense):
    """
    Detection defense based on applying the prevention defense on an attack image;
    and then comparing input image & filtered image
    ONLY on pixels considered by scaling algorithm (= more targeted than PrevBasedFilteringDetectionDefense).
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 prevention_defense_type: PreventionTypeDefense,
                 fourierpeakmatrixcollector: FourierPeakMatrixCollector,
                 quantile_value: float,
                 prevention_bandwidthfactor: int = 1
                 ):
        """
        Init detection method.
        :param verbose: show more information
        :param scaler_approach: scaling settings
        :param prevention_defense_type: specify type of prevention method {median_filter, random_filter}.
        :param quantile_value: necessary for defense setup, used in similary measurement.
        """
        super().__init__(verbose, scaler_approach)
        self.prevention_defense_type: PreventionTypeDefense = prevention_defense_type
        self.fourierpeakmatrixcollector: FourierPeakMatrixCollector = fourierpeakmatrixcollector
        self.prevention_bandwidthfactor: int = prevention_bandwidthfactor
        self.quantile_value: float = quantile_value

    def detect_attack_targeted_analysis(self,
                                        filtered_image: np.ndarray,
                                        att_image: np.ndarray,
                                        binary_mask: np.ndarray):

        filtered_image_vals = filtered_image[np.where(binary_mask == 1)].reshape(-1)
        input_image_vals = att_image[np.where(binary_mask == 1)].reshape(-1)

        val1 = np.abs(filtered_image_vals.astype(np.int) - input_image_vals.astype(np.int))
        score = np.quantile(val1, self.quantile_value)
        return score, {} # {'filtered_image': filtered_image}

    # @Overwrite
    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:

        preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=self.prevention_defense_type, scaler_approach=self.scaler_approach,
            fourierpeakmatrixcollector=self.fourierpeakmatrixcollector,
            bandwidth=self.prevention_bandwidthfactor, verbose_flag=False, usecythonifavailable=True
        )

        filtered_image = preventiondefense.make_image_secure(att_image=att_image)

        dir_attack_image = self.fourierpeakmatrixcollector.get(scaler_approach=self.scaler_approach)
        binary_mask_indices = np.where(dir_attack_image != 255)
        binary_mask = np.zeros((self.scaler_approach.cl_matrix.shape[1], self.scaler_approach.cr_matrix.shape[0]))
        binary_mask[binary_mask_indices] = 1

        score, score_dict = self.detect_attack_targeted_analysis(filtered_image=filtered_image,
                                                     att_image=att_image, binary_mask=binary_mask)

        if self.verbose:
            print("{}, {} got scores: {}".format(self.prevention_defense_type.name, self.prevention_bandwidthfactor,
                                                     score))

        return score, score_dict
