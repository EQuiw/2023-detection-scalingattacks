import enum

class AdaptiveDetectionStrategy(enum.Enum):
    """
    Adaptive strategies against detection methods for scaling attacks
    """
    jpeg = 1
    disable_frequencies = 2
    add_frequency_peak = 3
