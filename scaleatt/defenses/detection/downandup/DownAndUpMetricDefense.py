import numpy as np
import typing

from defenses.detection.downandup.DownAndUpScalingDefense import DownAndUpScalingDefense
from scaling.ScalingApproach import ScalingApproach
from utils.SimilarityMeasurementTool import SimilarityMeasurementTool
from utils.SimilarityMeasure import SimilarityMeasure


class DownAndUpMetricDefense(DownAndUpScalingDefense):
    """
    -- Down and Upscaling from Paper with Options {PSNR, MSE, SSIM} --

    Detection defense based on down- and upscaling, and then simply comparing both images
    using some (simple) metric.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 similarity_measurement: SimilarityMeasure
                 ):
        """
        Init detection method.
        :param verbose: show more information
        :param scaler_approach: scaling settings
        :param similarity_measurement: specify type of metric
        """
        super().__init__(verbose, scaler_approach)
        self.similarity_measurement: SimilarityMeasure = similarity_measurement

    # @Overwrite
    def compare_images(self, att_image: np.ndarray, output_upscaled: np.ndarray) -> typing.Tuple[float, dict]:

        score = SimilarityMeasurementTool.sim_measure(img1=att_image, img2=output_upscaled,
                                                      similarity_measurement=self.similarity_measurement)
        if self.verbose:
            print("{} got scores: {}".format(self.similarity_measurement.name, score))

        return score, {}

