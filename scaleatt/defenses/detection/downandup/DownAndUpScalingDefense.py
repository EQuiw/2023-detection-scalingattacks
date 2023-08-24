import numpy as np
from abc import abstractmethod
import typing

from defenses.detection.DetectionDefense import DetectionDefense
from scaling.ScalingApproach import ScalingApproach


class DownAndUpScalingDefense(DetectionDefense):
    """
    Generic upper class for
    detection defenses based on down- and upscaling,
    and then comparing the "original input" and its "down- & upscaled version".
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach):

        super().__init__(verbose, scaler_approach)


    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:
        # ensure that range is in [0,1]
        assert np.max(att_image) < 255.01 and np.min(att_image) >= -0.0001

        # downscale image under investigation
        result_output_image = self.scaler_approach.scale_image(xin=att_image)

        # upscale again
        src_shape0 = self.scaler_approach.cl_matrix.shape[1]
        src_shape1 = self.scaler_approach.cr_matrix.shape[0]

        output_upscaled = self.scaler_approach.scale_image_with(xin=result_output_image,
                                                                trows=src_shape0,
                                                                tcols=src_shape1)

        return self.compare_images(att_image=att_image, output_upscaled=output_upscaled)

    @abstractmethod
    def compare_images(self, att_image: np.ndarray, output_upscaled: np.ndarray) -> typing.Tuple[float, dict]:
        pass