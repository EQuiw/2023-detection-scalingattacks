import numpy as np
import typing

from utils.SimilarityMeasurementToolPatchBased import PatchMethod, compare_images_blockbased_targeted

from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector
from scaling.ScalingApproach import ScalingApproach
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from defenses.detection.filtering.PrevBasedFilteringTargetedDetectionDefense import PrevBasedFilteringTargetedDetectionDefense


class FilteringPreventionTargetedPatchBasedDefense(PrevBasedFilteringTargetedDetectionDefense):
    """
    -- Targeted Patch-Clean Filter from Paper --

    Filtering-based defense (prevention-based here) + patch-wise extraction
    ONLY on pixels considered by scaling algorithm (= more targeted than FilteringPreventionPatchBasedDefense).
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 prevention_defense_type: PreventionTypeDefense,
                 fourierpeakmatrixcollector: FourierPeakMatrixCollector,
                 quantile_value: float,
                 patch_comparison_params: dict,
                 patch_method: PatchMethod,
                 prevention_bandwidthfactor: int = 1,
                 ):
        """
        Init detection method.
        """
        super().__init__(verbose=verbose, scaler_approach=scaler_approach,
                         prevention_defense_type=prevention_defense_type,
                         fourierpeakmatrixcollector=fourierpeakmatrixcollector,
                         quantile_value=quantile_value,
                         prevention_bandwidthfactor=prevention_bandwidthfactor)

        # Params for patch-extraction part
        self.patch_method: PatchMethod = patch_method
        self.patch_comparison_params: dict = patch_comparison_params
        assert self.patch_method == PatchMethod.use_block_based, \
            "FilteringPreventionTargetedPatchBasedDefense only with block-based"


    # @Overwrite
    def detect_attack_targeted_analysis(self,
                                        filtered_image: np.ndarray,
                                        att_image: np.ndarray,
                                        binary_mask: np.ndarray):

        assert binary_mask.shape == filtered_image.shape[:2] == att_image.shape[:2]
        patch_scores: np.ndarray = compare_images_blockbased_targeted(img1=filtered_image, img2=att_image,
                                                                            quantile_value=self.quantile_value,
                                                                            binary_mask=binary_mask,
                                                                            **self.patch_comparison_params)

        # Aggregate scores into a single score value
        # score = np.abs(np.max(patch_scores) - np.mean(patch_scores))
        score = np.max(patch_scores) - np.mean(patch_scores)
        return score, {'patch_scores': (np.mean(patch_scores), np.min(patch_scores), np.max(patch_scores))}

