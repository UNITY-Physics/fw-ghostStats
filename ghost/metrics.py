import numpy as np

def calc_psnr(img1, img2, mask):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
    img1 (ndarray or ants image): The first image.
    img2 (ndarray or ants image): The second image.
    mask (ndarray or ants image): The mask indicating the region of interest.

    Returns:
    float: The PSNR value.

    """
    I1 = img1[mask==1]
    I2 = img2[mask==1]
    MSE = np.sum((I1-I2)**2)/len(I1)
    R = np.mean([max(I1),max(I2)])
    psnr = 10*np.log10(R**2/MSE)
    return psnr