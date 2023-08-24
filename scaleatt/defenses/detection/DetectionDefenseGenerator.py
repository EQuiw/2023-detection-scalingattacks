from scaling.ScalingApproach import ScalingApproach

from defenses.detection.DetectionDefense import DetectionDefense
from defenses.detection.DetectionTypeDefense import DetectionTypeDefense
from defenses.detection.frequency.CSPFrequencyDefense import CSPFrequencyDefense
from defenses.detection.frequency.FourierSpectrumSamplingDefense import FourierSpectrumSamplingDefense
from defenses.detection.frequency.FourierSpectrumDistanceDefense import FourierSpectrumDistanceDefense
from defenses.detection.downandup.DownAndUpMetricDefense import DownAndUpMetricDefense
from defenses.detection.filtering.FilteringDetectionDefense import FilteringDetectionDefense
from defenses.detection.filtering.PrevBasedFilteringDetectionDefense import PrevBasedFilteringDetectionDefense
from defenses.detection.filtering.PrevBasedFilteringTargetedDetectionDefense import PrevBasedFilteringTargetedDetectionDefense
from defenses.detection.downandup.HistogramScatteringDefense import HistogramScatteringDefense

from defenses.detection.filtering.PatchBased.FilteringPreventionPatchBasedDefense import FilteringPreventionPatchBasedDefense
from defenses.detection.filtering.PatchBased.FilteringPreventionTargetedPatchBasedDefense import FilteringPreventionTargetedPatchBasedDefense


class DetectionDefenseGenerator:
    """
    Generator for various detection defenses.
    """

    @staticmethod
    def create_detection_defense(defense_type: DetectionTypeDefense,
                                 verbose_flag: bool,
                                 scaler_approach: ScalingApproach,
                                 **kwargs) -> DetectionDefense:

        if defense_type == DetectionTypeDefense.frequency_csp:
            return CSPFrequencyDefense(verbose=verbose_flag,
                                       scaler_approach=scaler_approach,
                                       **kwargs)
        elif defense_type == DetectionTypeDefense.frequency_spectrum_sampling:
            return FourierSpectrumSamplingDefense(verbose=verbose_flag,
                                                  scaler_approach=scaler_approach,
                                                  **kwargs)
        elif defense_type == DetectionTypeDefense.frequency_spectrum_distance:
            return FourierSpectrumDistanceDefense(verbose=verbose_flag,
                                                  scaler_approach=scaler_approach)
        elif defense_type == DetectionTypeDefense.downandup:
            return DownAndUpMetricDefense(verbose=verbose_flag,
                                          scaler_approach=scaler_approach,
                                          **kwargs)
        elif defense_type == DetectionTypeDefense.downandup_histoscattering:
            return HistogramScatteringDefense(verbose=verbose_flag,
                                              scaler_approach=scaler_approach,
                                              **kwargs)
        elif defense_type == DetectionTypeDefense.filtering_min or \
            defense_type == DetectionTypeDefense.filtering_max:
            return FilteringDetectionDefense(verbose=verbose_flag,
                                             scaler_approach=scaler_approach,
                                             detection_method=defense_type,
                                             **kwargs)
        elif defense_type == DetectionTypeDefense.filtering_prevention:
            return PrevBasedFilteringDetectionDefense(verbose=verbose_flag,
                                                      scaler_approach=scaler_approach,
                                                      **kwargs)
        elif defense_type == DetectionTypeDefense.filtering_targeted_prevention:
            return PrevBasedFilteringTargetedDetectionDefense(verbose=verbose_flag,
                                                      scaler_approach=scaler_approach,
                                                      **kwargs)
        elif defense_type == DetectionTypeDefense.filtering_patch_prevention_block:
            return FilteringPreventionPatchBasedDefense(verbose=verbose_flag,
                                                        scaler_approach=scaler_approach,
                                                        **kwargs)
        elif defense_type == DetectionTypeDefense.filtering_targeted_patch_prevention_block:
            return FilteringPreventionTargetedPatchBasedDefense(verbose=verbose_flag,
                                                        scaler_approach=scaler_approach,
                                                        **kwargs)
        else:
            raise NotImplementedError("Passed detection type not implemented in generator, yet")
