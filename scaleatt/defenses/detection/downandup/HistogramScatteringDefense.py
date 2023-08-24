import numpy as np
from enum import Enum
import typing
from scipy import spatial
import cv2 as cv
import matplotlib.pyplot as plt

from defenses.detection.downandup.DownAndUpScalingDefense import DownAndUpScalingDefense
from scaling.ScalingApproach import ScalingApproach
from utils.plot_image_utils import plot_images_in_actual_size


class UsenixDefenseChoice(Enum):
    """
    Choice what defense from Xiao et al. should be used.
    """
    use_histogram = 1
    use_scattering = 2





class HistogramScatteringDefense(DownAndUpScalingDefense):
    """
    -- Down and Upscaling from Paper with Options {Histogram, Color-scattering} --

    Detection defense based on histogram or color scattering comparison between att-img
    and upscaled out-img.
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 method: UsenixDefenseChoice
                 ):
        super().__init__(verbose, scaler_approach)
        self.method: UsenixDefenseChoice = method


    # @Overwrite
    def compare_images(self, att_image: np.ndarray, output_upscaled: np.ndarray) -> typing.Tuple[float, dict]:

        if self.method == UsenixDefenseChoice.use_histogram:
            return self.do_histogram_comparison(att_image=att_image, output_upscaled=output_upscaled), {}
        elif self.method == UsenixDefenseChoice.use_scattering:
            return self.do_scattering_comparison(att_image=att_image, output_upscaled=output_upscaled), {}
        else:
            raise NotImplementedError()


    def do_histogram_comparison(self, att_image: np.ndarray, output_upscaled: np.ndarray) -> float:

        # first convert both images to grayscale
        att_image_gray = cv.cvtColor(att_image, cv.COLOR_RGB2GRAY)
        output_upscaled_gray = cv.cvtColor(output_upscaled, cv.COLOR_RGB2GRAY)

        # get histo.
        counts_att, bins_att = np.histogram(att_image_gray, range(257))
        counts_out, bins_out = np.histogram(output_upscaled_gray, range(257))

        if self.verbose is True:
            plot_images_in_actual_size(imgs=[att_image_gray, output_upscaled_gray],titles=['Att','Upsc.'],rows=1)
            plt.bar(bins_att[:-1] - 0.5, counts_att, width=1, edgecolor='none',alpha=0.5)
            plt.bar(bins_out[:-1] - 0.5, counts_out, width=1, edgecolor='none',alpha=0.5)
            plt.xlim([-0.5, 255.5])
            plt.show()

        cos_similarity = spatial.distance.cosine(counts_att, counts_out)
        return 1-cos_similarity


    def do_scattering_comparison(self, att_image: np.ndarray, output_upscaled: np.ndarray) -> float:

        # first convert both images to grayscale
        att_image_gray = cv.cvtColor(att_image, cv.COLOR_RGB2GRAY)
        output_upscaled_gray = cv.cvtColor(output_upscaled, cv.COLOR_RGB2GRAY)

        xcenter = int(att_image.shape[0] / 2)
        ycenter = int(att_image.shape[1] / 2)

        def scattering(img):
            counts = np.zeros(256)
            for ip in range(256):
                pos_matches = np.where(img == ip)
                if len(pos_matches[0]) > 1:
                    # l2 distance:
                    avg_dist = np.sqrt(np.square(pos_matches[0] - xcenter) + np.square(pos_matches[1] - ycenter))
                    # manhatten distance:
                    # avg_dist = (np.abs(pos_matches[0] - xcenter) + np.abs(pos_matches[1] - ycenter))
                    counts[ip] = avg_dist.mean()
            return counts

        counts_att = scattering(img=att_image_gray)
        counts_out = scattering(img=output_upscaled_gray)

        if self.verbose is True:
            plot_images_in_actual_size(imgs=[att_image_gray, output_upscaled_gray],titles=['Att','Upsc.'],rows=1)
            # bar chart
            # plt.bar(np.arange(256) - 0.5, counts_att, width=1, edgecolor='none',alpha=0.5)
            # plt.bar(np.arange(256) - 0.5, counts_out, width=1, edgecolor='none',alpha=0.5)
            # plt.xlim([-0.5, 255.5])
            # plt.show()

            # bar chart, but with just lines
            plt.step(np.arange(256), counts_att)
            plt.step(np.arange(256), counts_out)
            plt.xlim([0, 255])
            plt.show()


        cos_similarity = spatial.distance.cosine(counts_att, counts_out)
        return 1 - cos_similarity