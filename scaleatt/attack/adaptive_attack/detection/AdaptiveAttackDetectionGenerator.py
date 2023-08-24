from scaling.ScalingApproach import ScalingApproach

from attack.adaptive_attack.detection.AdaptiveDetectionStrategy import AdaptiveDetectionStrategy
from attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage2
from attack.adaptive_attack.detection.AdaptiveDetectionJPEGAttack import AdaptiveDetectionJPEGAttack
from attack.adaptive_attack.detection.frequency.DisableFrequenciesStrategy import DisableFrequenciesStrategy
from attack.adaptive_attack.detection.frequency.AddFrequencyPeakStrategy import AddFrequencyPeakStrategy



class AdaptiveAttackDetectionGenerator:
    """
    Generator for various adaptive attacks against detection defenses
    """

    @staticmethod
    def create_adaptive_attack(attack_type: AdaptiveDetectionStrategy,
                               verbose_flag: bool,
                               scaler_approach: ScalingApproach,
                               **kwargs) -> AdaptiveAttackOnAttackImage2:
        """
        Creates a specific adaptive attack against defense.

        Attack_params:
        - AdaptiveDetectionStrategy.jpeg requires 'args_adaptive_attack_jpeg_quality'
        """

        if attack_type == AdaptiveDetectionStrategy.jpeg:
            adaptiveattack: AdaptiveDetectionJPEGAttack = AdaptiveDetectionJPEGAttack(verbose=verbose_flag,
                                                                                      scaler_approach=scaler_approach,
                                                                                      **kwargs)
        elif attack_type == AdaptiveDetectionStrategy.disable_frequencies:
            adaptiveattack: DisableFrequenciesStrategy = DisableFrequenciesStrategy(verbose=verbose_flag,
                                                                                    scaler_approach=scaler_approach,
                                                                                    **kwargs)
        elif attack_type == AdaptiveDetectionStrategy.add_frequency_peak:
            adaptiveattack: AddFrequencyPeakStrategy = AddFrequencyPeakStrategy(verbose=verbose_flag,
                                                                                scaler_approach=scaler_approach,
                                                                                **kwargs)
        else:
            raise NotImplementedError("Passed adaptive attack type not implemented in generator, yet")

        return adaptiveattack
