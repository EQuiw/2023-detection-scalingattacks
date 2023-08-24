import numpy as np
import typing

from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector
from defenses.detection.DetectionDefense import DetectionDefense
from scaling.ScalingApproach import ScalingApproach
from utils.SimilarityMeasurementTool import SimilarityMeasurementTool
from utils.SimilarityMeasure import SimilarityMeasure
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefense import PreventionDefense


class PrevBasedFilteringDetectionDefense(DetectionDefense):
    """
    -- Clean-Filter from Paper with Options: {Median-Filter, Random-Filter} + {PSNR, SSIM} --

    Detection defense based on applying the prevention defense on an attack image;
    and then comparing input image & filtered image.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 prevention_defense_type: PreventionTypeDefense,
                 fourierpeakmatrixcollector: FourierPeakMatrixCollector,
                 similarity_measurement: SimilarityMeasure,
                 prevention_bandwidthfactor: int = 1
                 ):
        """
        Init detection method.
        :param verbose: show more information
        :param scaler_approach: scaling settings
        :param prevention_defense_type: specify type of prevention method {median_filter, random_filter}.
        :param similarity_measurement: specify type of metric
        """
        super().__init__(verbose, scaler_approach)
        self.prevention_defense_type: PreventionTypeDefense = prevention_defense_type
        self.similarity_measurement: SimilarityMeasure = similarity_measurement
        self.fourierpeakmatrixcollector: FourierPeakMatrixCollector = fourierpeakmatrixcollector
        self.prevention_bandwidthfactor: int = prevention_bandwidthfactor

    # @Overwrite
    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:

        preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=self.prevention_defense_type, scaler_approach=self.scaler_approach,
            fourierpeakmatrixcollector=self.fourierpeakmatrixcollector,
            bandwidth=self.prevention_bandwidthfactor, verbose_flag=False, usecythonifavailable=True
        )

        filtered_image = preventiondefense.make_image_secure(att_image=att_image)

        score = SimilarityMeasurementTool.sim_measure(img1=att_image, img2=filtered_image,
                                                      similarity_measurement=self.similarity_measurement)
        if self.verbose:
            print("{}, {}, {} got scores: {}".format(self.prevention_defense_type.name, self.prevention_bandwidthfactor,
                                                     self.similarity_measurement.name, score))

        return score, {} # {'filtered_image': filtered_image}
