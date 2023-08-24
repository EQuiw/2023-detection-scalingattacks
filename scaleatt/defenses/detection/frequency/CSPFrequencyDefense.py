import numpy as np
import typing
import tempfile

from defenses.detection.frequency.FrequencyDefense import FrequencyDefense
from scaling.ScalingApproach import ScalingApproach

import cv2


class CSPFrequencyDefense(FrequencyDefense):
    """
    -- CSP and CSP-improved from Paper --

    Counting CSP in frequency spectrum based on Kim et al.
    We implement four methods:
    1 - contours1, used directly (CSP-improved, version 1)
    2 - contours2, used directly (CSP-improved, version 2)
    13 - contours1, but checked if #number > 1 (related to paper, CSP, version 1)
    23 - contours2, but checkef if #number > 1 (related to paper, CSP, version 2)

    Note that we evaluated all methods, but only presented method 1 and 13 in the paper (version 1 of CSP-improved
    and CSP, respectively).
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 contourmethod: int):

        super().__init__(verbose, scaler_approach)
        self.contourmethod: int = contourmethod
        assert self.contourmethod in [1,2,13,23], "contourmethod param needs to be 1, 2, 13, or 23"

    def detect_attack(self, att_image: np.ndarray) -> typing.Tuple[float, dict]:
        contours1, contours2 = self._get_contours(input_image=att_image)
        if self.contourmethod == 1:
            return contours1, {} # {'contours2': contours2}
        elif self.contourmethod == 2:
            return contours2, {} # {'contours1': contours1}
        elif self.contourmethod == 13:
            return float(contours1 > 1), {}
        elif self.contourmethod == 23:
            return float(contours2 > 1), {}
        else:
            raise NotImplementedError()

    def _get_contours(self, input_image) -> typing.Tuple[float, float]:
        # Code as from repository by Kim et al.; including the imwrite and imreads
        # to rule out that a change of API changes the defense capability.
        temp_path_for_saving = tempfile.TemporaryDirectory(prefix="csp_frequency_defense")

        save_path: str = str(temp_path_for_saving.name + "/" + "input.png")
        cv2.imwrite(save_path, input_image)
        original_3 = cv2.imread(save_path, 2) # Option 2 can make output grayscale
        # original_3 = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

        f = np.fft.fft2(original_3)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        filename = str(temp_path_for_saving.name + "/" + ('test_or_mag_' + '.jpg'))
        w=cv2.imwrite(filename, magnitude_spectrum)
        img = cv2.imread(filename)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Function threshold implements Binary Threshold with cv2.threshold(src, thresh, maxValue):
        # "if src(x,y) > thresh: dst(x,y) = maxValue else: dst(x,y) = 0"
        ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)


        medianFiltered = cv2.medianBlur(thresh, 3)
        blur = cv2.GaussianBlur(medianFiltered, (21, 21), 0)

        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        q = cv2.drawContours(img, contours, -1, (0, 0, 255), 5)


        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret2, thresh2 = cv2.threshold(gray2,250,255,cv2.THRESH_BINARY)
        ret2, thresh2 = cv2.threshold(gray2, 240, 255, cv2.THRESH_BINARY)

        medianFiltered2 = cv2.medianBlur(thresh2, 3)
        # medianFiltered2_2 = cv2.medianBlur(medianFiltered2, 7)
        blur2 = cv2.GaussianBlur(medianFiltered2, (21, 21), 0)

        # cv2.imshow('Binary image of origin_magnitude ',medianFiltered2)
        # cv2.waitKey(0)

        # contours2, hierarchy2 = cv2.findContours(medianFiltered2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(blur2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        q2 = cv2.drawContours(img, contours2, -1, (0, 0, 255), 5)

        #plt.imshow(q2)
        #plt.show()

        # if (len(contours) == 1) & (len(contours2) == 3):
        #     cnt2 += 1
        # if (len(contours) == 1):
        #     cnt_contours += 1
        #     if (len(contours2) == 3):
        #         cnt += 1
        #
        # if (len(contours2) == 3):
        #     cnt_contours2 += 1

        if self.verbose:
            print("contours:", len(contours), len(contours2))
        return float(len(contours)), float(len(contours2))