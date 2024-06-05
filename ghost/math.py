import numpy as np
from scipy.special import i0e

def logi0e(x):
    """
    Calculates the logarithm of the exponentially scaled modified Bessel function of the first kind (i0e) plus x.

    Parameters:
    x (float): The input value.

    Returns:
    float: The result of np.log(i0e(x)) + x.
    """

    return np.log(i0e(x)) + x

def rician_loglike(x, sigma, mu):
    """
    Calculate the log-likelihood of a Rician distribution.

    Parameters:
    x (float): The observed value.
    sigma (float): The scale parameter of the Rician distribution.
    mu (float): The location parameter of the Rician distribution.

    Returns:
    float: The log-likelihood of the Rician distribution.

    """
    return np.log(x) - 2*np.log(sigma) - (x**2 + mu**2)/(2*sigma**2) + logi0e(x*mu/sigma**2)

def make_sphere(img_shape, radius, center):
    """
    Create a binary sphere mask with the given radius and center.

    Parameters:
    img_shape (tuple): The shape of the output mask in the form (nx, ny, nz).
    radius (float): The radius of the sphere.
    center (tuple): The center coordinates of the sphere in the form (cx, cy, cz).

    Returns:
    numpy.ndarray: A binary mask representing the sphere.
    """

    nx, ny, nz = img_shape
    
    cx = nx - center[0] - nx//2
    cy = ny - center[1] - ny//2
    cz = nz - center[2] - nz//2
    
    X, Y, Z = np.meshgrid(np.arange(-ny//2, ny//2) + cy, np.arange(-nx//2, nx//2) + cx, np.arange(-nz//2, nz//2) + cz)
    D = np.sqrt(X**2 + Y**2 + Z**2)
    
    sphere = np.zeros(img_shape)
    sphere[D < radius] = 1

    return sphere

def make_circle(img_shape, radius, center):
    """
    Creates a binary circle image with the specified radius and center.

    Parameters:
    img_shape (tuple): The shape of the output image (height, width).
    radius (float): The radius of the circle.
    center (tuple): The center coordinates of the circle (x, y).

    Returns:
    numpy.ndarray: The binary circle image.
    """

    nx, ny = img_shape
    X, Y = np.meshgrid(np.arange(-nx//2, nx//2) + center[0], np.arange(-ny//2, ny//2) + center[1])
    D = np.sqrt(X**2 + Y**2)
    circle = np.zeros(img_shape)
    circle[D < radius] = 1
    return circle
