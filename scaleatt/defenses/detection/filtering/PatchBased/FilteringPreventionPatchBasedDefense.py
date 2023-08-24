import numpy as np
import typing
import cv2 as cv

from utils.SimilarityMeasurementToolPatchBased import sim_measure_patchwise, PatchMethod

from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector
from defenses.detection.DetectionDefense import DetectionDefense
from scaling.ScalingApproach import ScalingApproach
from utils.SimilarityMeasure import SimilarityMeasure
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefense import PreventionDefense


class FilteringPreventionPatchBasedDefense(DetectionDefense):
    """
    -- Patch-Clean Filter from Paper --

    Filtering-based defense (prevention-based here) + patch-wise extraction.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 prevention_defense_type: PreventionTypeDefense,
                 fourierpeakmatrixcollector: FourierPeakMatrixCollector,
                 similarity_measurement: SimilarityMeasure,
                 patch_comparison_params: dict,
                 patch_method: PatchMethod,
                 prevention_bandwidthfactor: int = 1,
                 ):
        """
        Init detection method.
        """
        super().__init__(verbose, scaler_approach)

        # Params for prevention-based part
        self.prevention_defense_type: PreventionTypeDefense = prevention_defense_type
        self.fourierpeakmatrixcollector: FourierPeakMatrixCollector = fourierpeakmatrixcollector
        self.prevention_bandwidthfactor: int = prevention_bandwidthfactor

        # Params for patch-extraction part
        self.patch_method: PatchMethod = patch_method
        self.patch_comparison_params: dict = patch_comparison_params

        # Params for both parts
        self.similarity_measurement: SimilarityMeasure = similarity_measurement

    # @Overwrite
    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:

        preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=self.prevention_defense_type, scaler_approach=self.scaler_approach,
            fourierpeakmatrixcollector=self.fourierpeakmatrixcollector,
            bandwidth=self.prevention_bandwidthfactor, verbose_flag=False, usecythonifavailable=True
        )

        filtered_image = preventiondefense.make_image_secure(att_image=att_image)
        small_filtered_image = self.scaler_approach.scale_image(filtered_image)

        small_result_attack_image = self.scaler_approach.scale_image(att_image)
        small_result_attack_image = cv.GaussianBlur(small_result_attack_image, (3, 3), 0)

        return self.compare_images(img1=small_result_attack_image, img2=small_filtered_image)

    def compare_images(self, img1: np.ndarray, img2: np.ndarray):
        return self._compare_images_patchbased(img1=img1, img2=img2,
                                               patch_method=self.patch_method,
                                               patch_params=self.patch_comparison_params)

    # @Overwrite
    def _compare_images_patchbased(self,
                                   img1: np.ndarray,
                                   img2: np.ndarray,
                                   patch_method: PatchMethod,
                                   patch_params: dict) -> typing.Tuple[float, dict]:

        # Compute scores per patch
        patch_scores: np.ndarray = sim_measure_patchwise(patch_method=patch_method,
                                                         img1=img1, img2=img2,
                                                         similarity_measurement=self.similarity_measurement,
                                                         **patch_params)

        # Aggregate scores into a single score value
        score = np.abs(np.mean(patch_scores) - np.min(patch_scores))

        if self.verbose:
            print("{} got scores: {}".format(self.similarity_measurement.name, score))

        return score, {'patch_scores': (np.mean(patch_scores), np.min(patch_scores))}
