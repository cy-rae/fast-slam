import numpy as np
from numpy import ndarray
from scipy import ndimage


class LineFilter:
    """
    The LineFilter class is used to filter points to reduce noise.
    """

    @staticmethod
    def filter(points: ndarray, sigma=0.1):
        """
        Apply a Gaussian filter to the points to reduce noise.
        :param points: The points to filter
        :param sigma: The standard deviation of the Gaussian filter. Default is 0.1.
        :return: Returns the filtered points.
        """
        x_filtered = ndimage.gaussian_filter1d(points[:, 0], sigma=sigma)
        y_filtered = ndimage.gaussian_filter1d(points[:, 1], sigma=sigma)
        return np.column_stack((x_filtered, y_filtered))
