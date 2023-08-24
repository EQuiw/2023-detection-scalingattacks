import enum

class DetectionTypeDefense(enum.Enum):
    """
    List of all detection types
    """
    frequency_spectrum_distance = 1
    frequency_spectrum_sampling = 2
    downandup = 5
    downandup_histoscattering = 6

    frequency_csp = 10
    filtering_min = 15
    filtering_max = 16
    filtering_prevention = 17
    filtering_targeted_prevention = 18
    filtering_patch_prevention_block = 20
    filtering_targeted_patch_prevention_block = 21