import skimage
import numpy as np
from numba import njit


"""
Adapted from the imagecorruptions library which uses older version of skimage where gaussian is applied with multichannel arg.
It has been replaced with channel_axis
"""

def wrapper_gaussian_filter(image, sigma, multichannel=False, channel_axis=-1):
    """
    Apply Gaussian filter to the input image.

    Parameters:
    - image (ndarray): Input image.
    - sigma (float or sequence of floats): Standard deviation(s) for Gaussian kernel.
    - multichannel (bool, optional): To be used for scikit-image version < 0.19
    - channel_axis (int, optional): To be used for scikit-image version >= 0.19
    Returns:
    - ndarray: Image after Gaussian filtering.

    Notes:
    - If the scikit-image version is 0.19 or higher, the 'channel_axis' parameter is used instead of 'multichannel'.
    - Otherwise, 'multichannel' parameter is used for filtering.

    """
    if skimage.__version__ >= '0.19':
        # Use channel_axis instead of multichannel
        return skimage.filters.gaussian(image, sigma, channel_axis=channel_axis)
    else:
        # Use multichannel parameter
        return skimage.filters.gaussian(image, sigma, multichannel=multichannel)

@njit()
def _shuffle_pixels_njit_glass_blur(d0,d1,x,c):
    """
    Apply pixel shuffling for glass blur effect.

    Parameters:
    - d0 (int): Height of the image.
    - d1 (int): Width of the image.
    - x (ndarray): Input image.
    - c (tuple): Parameters for shuffling.

    Returns:
    - ndarray: Image after pixel shuffling for glass blur effect.
    """

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x

def gaussian_blur(x, severity=1):
    """
    Apply Gaussian blur to the input image.
    
    Parameters:
    - x (ndarray): Input image.
    - severity (int, optional): Severity level of the blur effect. Defaults to 1.
    
    Returns:
    - ndarray: Image after Gaussian blur effect.
    """

    c = [1, 2, 3, 4, 6][severity - 1]
    x = wrapper_gaussian_filter(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255
    
def glass_blur(x, severity=1):
    """
    Apply glass blur effect to the input image.

    Parameters:
    - x (ndarray): Input image.
    - severity (int, optional): Severity level of the blur effect. Defaults to 1.

    Returns:
    - ndarray: Image after glass blur effect.

    """
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(wrapper_gaussian_filter(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
    x = _shuffle_pixels_njit_glass_blur(np.array(x).shape[0],np.array(x).shape[1],x,c)

    return np.clip(wrapper_gaussian_filter(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255