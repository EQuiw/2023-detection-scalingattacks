import numpy as np
import tempfile
from typing import Tuple
import cv2 as cv

from utils.save_image_data import save_jpeg_image
from attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage2
from scaling.ScalingApproach import ScalingApproach


class AdaptiveDetectionJPEGAttack(AdaptiveAttackOnAttackImage2):
    """
    Adaptive attack against detection defense by simply compressing an image.
    Creates a JPEG version; internally, it will save the image on a temporary directory to this end.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 quality_factor: int
                 ):
        """

        :param verbose: print debug messages
        :param scaler_approach: scaler approach
        :param quality_factor: quality factor of JPEG compression (0-100)
        """
        super().__init__(verbose, scaler_approach)
        self.quality_factor = quality_factor

    # @Overwrite
    def counter_attack(self, att_image: np.ndarray, src: np.ndarray, tar: np.ndarray) -> Tuple[np.ndarray, dict]:
        cur_attimg = att_image
        temp_path_for_saving = tempfile.TemporaryDirectory(prefix="eval_detection_jpeg")
        save_path: str = str(temp_path_for_saving.name + "/" + "attack_image.jpg")
        save_jpeg_image(out_img=cur_attimg, image_path=save_path, quality=self.quality_factor)
        cur_attimg_jpeg = cv.imread(save_path)
        cur_attimg_jpeg = cv.cvtColor(cur_attimg_jpeg, cv.COLOR_BGR2RGB)
        temp_path_for_saving.cleanup()
        return cur_attimg_jpeg, {}
