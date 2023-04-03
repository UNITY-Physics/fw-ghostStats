from . import GHOSTDIR
import subprocess as sp
import os
import pandas as pd
import ants
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

"""
Functions to deal with the phantom images
"""

def calculate_slice_thickness_from_wedges(img_data, seg_data, sigma=3, wedge_angle=10, resolution=None, return_plot_data=False):
    """Calculate the slice thickness of a 3D image. The slice thickness is calculated by fitting a Gaussian to the intensity gradient along the wedges of the image and calculating the slice thickness from the standard deviation of the Gaussian.
    
    Parameters
    ----------
    nifti_image : nifti image or numpy array
        The 3D image to calculate the slice thickness of.

    nifti_segmentation : nifti image or numpy array
        The segmentation of the 3D image indicating the location of the wedges. The segmentation of the wedges should be labelled 1 and 2.

    sigma : int, optional
        The standard deviation of the Gaussian window used to smooth the data. The default is 3.

    wedge_angle : int, optional
        The angle of the wedge in degrees. The default is 10.

    resolution : int, optional
        The resolution of the image in mm. Neccessary if image data is numpy.array The default is None.

    return_plot_data : bool, optional
        Whether to return the data needed to plot the slice thickness distributions. The default is False.
    
    Returns
    -------
    slice_thickness : int
        The slice thickness of the 3D image.

    w : int
        The projected wedge thicknesses.

    theta : int
        The angle of the image plane with respect to the base of the wedges.

    plot_data : dict
        The data needed to plot the slice thickness distributions. Only returned if return_plot_data is True.
    """
    
    if isinstance(img_data, (nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image)):
        # Get the image data and resolution
        image = img_data.get_fdata()
        head = img_data.header
        resolution = head.get('pixdim')[1:4]
    elif isinstance(img_data, np.ndarray):
        # Get the image data and resolution
        image = img_data
        if resolution is None:
            raise TypeError("Resolution is not defined. Please define the resolution of the image in mm.")
    if isinstance(seg_data, (nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image)):
        # Get the wedge segmentations
        seg1 = seg_data.get_fdata() == 1
        seg2 = seg_data.get_fdata() == 2
    elif isinstance(seg_data, np.ndarray):
        seg1 = seg_data == 1
        seg2 = seg_data == 2

    intensities = [] # TK better variable name
    for seg in [seg1, seg2]:
        id = np.where(seg)
        min_i, max_i = np.min(id[0]), np.max(id[0])
        min_j, max_j = np.min(id[1]), np.max(id[1])
        min_k, max_k = np.min(id[2]), np.max(id[2])
        yz_planes = image[min_i:max_i+1, min_j:max_j+1, min_k:max_k+1] # yz
        slice_id = np.arange(np.min(id[2]),np.max(id[2])+1)
        intensities.append(np.mean(yz_planes, axis=0))

    indices = []
    for intensity in intensities:
        mean_intensity_y = np.mean(intensity, axis=0)
        index_mean = np.argmin(np.abs(intensity - mean_intensity_y), axis=0)
        percentage_mean = (index_mean / len(intensity)) * 100
        indices.append(np.argmin(np.abs(50 - percentage_mean)))
    
    # Make sure the wedges are segmented correctly
    if indices[0] == indices[1]:
        index = indices[0]
    else:
        raise TypeError("Index of wedges do not match. Check segmentation of the wedges.")
    
    slice = slice_id[index]

    lp1, lp2 = intensities
    lp = np.stack((lp1[:,index],lp2[:,index])) # line profile

    x_axis = np.arange(lp.shape[1])*resolution[1]

    # Filter the data with a Gaussian window to make it differentiable
    lp_smooth = ndi.gaussian_filter1d(lp, sigma, 1)

    # Differentiate the data with respect to image position
    lp_diff = np.abs(np.gradient(lp_smooth, axis=1))

    # define the gaussian function to fit
    def gauss(x, a, b, c):
        return a * np.exp(-((x-b)**2/(2*c**2)))
    
    lp_fit = np.empty((lp_diff.shape[0], lp_diff.shape[1]))
    w = np.empty(2) # projected wedge thicknesses
    for i in range(lp_diff.shape[0]):
        # Perform the curve fit
        popt, pcov = curve_fit(gauss, x_axis, lp_diff[i], p0=[1,0,1], bounds=([0,-np.inf,0], [np.inf, np.inf, np.inf]))
        
        # Calculate the fitted curves
        lp_fit[i] = gauss(x_axis, *popt)

        # Compute the projected wedge thicknesses
        w[i] = popt[2]*2*np.sqrt(2*np.log(2))

    # Calculate theta (the angle of the image plane with respect to the base of the wedges)
    theta = 1 / 2 * np.arcsin((w[1] - w[0]) * np.sin(2 * np.radians(10)) / sum(w))

    # Get the slice thickness
    slice_thickness = w[0]  * np.tan(np.radians(wedge_angle) + theta)

    if return_plot_data:
        # Make dictionary of variables needed to plot the slice thickness distributions
        plot_data = {'lp': lp, 
                     'lp_smooth': lp_smooth, 
                     'lp_diff': lp_diff, 
                     'lp_fit': lp_fit, 
                     'x_axis': x_axis, 
                     'seg1': seg1, 
                     'seg2': seg2, 
                     'slice': slice}
        return slice_thickness, w, theta, plot_data
    else:
        return slice_thickness, w, theta

def transform_reference_segmentation(ref_img_path, ref_seg_path, target_img_path):
    """Register the target image to the reference image and apply the transformation to the reference segmentation.
    
    Parameters
    ----------
    ref_img_path : str
        Path to the reference image.
    
    ref_seg_path : str
        Path to the reference segmentation.
        
    target_img_path : str
        Path to the target image.
        
    Returns
    -------
    img_data : numpy.ndarray
        The target image data.
    
    seg_data : numpy.ndarray
        The transformed reference segmentation."""

    ref = ants.image_read(ref_img_path)
    seg = ants.image_read(ref_seg_path)
    target = ants.image_read(target_img_path)

    reg = ants.registration(fixed=ref, moving=target, type_of_transform='Affine')
    seg_warp = ants.apply_transforms(fixed=target, moving=seg, whichtoinvert=[1], transformlist=reg['fwdtransforms'], interpolator='genericLabel')

    img_data = target.numpy()
    seg_data = seg_warp.numpy()

    return img_data, seg_data

def get_phantom_nii(weighting='T1'):
    """Get filename of phantom image

    Args:
        weighting (str, optional): Which weighting (T1 or T2). Default is 'T1'.

    Raises:
        ValueError: Wrong weighting

    Returns:
        str: Full file path
    """
    avail_weightings = ['T1', 'T2']
    if weighting not in avail_weightings:
        raise ValueError(f'Not a valid weighting. (Valid: {avail_weightings})')
    else:
        return os.path.join(GHOSTDIR, 'data', f'{weighting}_phantom.nii.gz')

def get_seg_nii(seg='T1'):
    """Get filename of segmentation image

    Args:
        seg (str, optional): Which segmentation (T1, T2, ADC, LC, fiducials, wedges). Default is 'T1'.

    Raises:
        ValueError: Wrong segmentation

    Returns:
        str: Full file path
    """
    avail_seg = ['T1', 'T2', 'ADC', 'LC', 'wedges', 'fiducials']
    if seg not in avail_seg:
        raise ValueError(f'Not a valid segmentation. (Valid: {avail_seg})')
    else:
        if seg == 'T1' or seg == 'T2' or seg == 'ADC':
            return os.path.join(GHOSTDIR, 'data', f'{seg}_mimics.nii.gz')
        elif seg == 'fiducials' or seg == 'wedges':
            return os.path.join(GHOSTDIR, 'data', f'{seg}.nii.gz')
        elif seg == 'LC':
            return os.path.join(GHOSTDIR, 'data', f'{seg}_vials.nii.gz')
        
def download_ref_data():
    """
    Downloads reference data for the phantom from Dropbox
    """
    files = [{'fname':'T1_phantom.nii.gz',
            'link':'https://www.dropbox.com/s/cwujos81rtt6s87/T1_phantom.nii.gz?dl=0'},
            {'fname':'T2_phantom.nii.gz',
            'link':'https://www.dropbox.com/s/pcq7be6q019j6jb/T2_phantom.nii.gz?dl=0'},
            {'fname':'ADC_mimics.nii.gz',
            'link':'https://www.dropbox.com/s/yuf0sl9uz1bkqu5/ADC_mimics.nii.gz?dl=0'},
            {'fname':'LC_vials.nii.gz',
            'link':'https://www.dropbox.com/s/d11js9ct8wvms48/LC_vials.nii.gz?dl=0'},
            {'fname':'T2_mimics.nii.gz',
            'link':'https://www.dropbox.com/s/vkkwd02f8dz2nqu/T2_mimics.nii.gz?dl=0'},
            {'fname':'T1_mimics.nii.gz',
            'link':'https://www.dropbox.com/s/0ai6z3cg94xcvn8/T1_mimics.nii.gz?dl=0'},
            {'fname':'fiducials.nii.gz',
            'link':'https://www.dropbox.com/s/1e6dzar48ajx3zt/fiducials.nii.gz?dl=0'},
            {'fname':'wedges.nii.gz',
            'link':'https://www.dropbox.com/s/y9k852idyxi8vwa/wedges.nii.gz?dl=0'}
            ]

    # Check if folder exists
    dl_path = f"{GHOSTDIR}/data"
    if not os.path.exists(dl_path):
        os.mkdir(dl_path)
        print(f"Created folder: {dl_path}")

    for f in files:
        file_path = f"{dl_path}/{f['fname']}"
        if not os.path.exists(file_path):
            dl_link = f['link']
            print('Downloading %s from %s'%(f['fname'], f['link']))
            cmd = f'wget -q -O {file_path} "{dl_link}"'
            out = sp.call(cmd, shell=True)
            print(f'Done. File saved to {file_path}')

        else:
            print("%s is already downloaded. Skipping"%f['fname'])

def reg_to_phantom(target_img, phantom_weighting='T1', xfm_type='Affine'):
    """Get transformation object from target image to reference image
    
    Parameters
    ----------
    target_img : antsImage
        The target image. Probably from the swoop.
    
    phantom_weighting : str
        Which weighting (T1 or T2). Default is 'T1'.

    xfm_type : str
        The type of transformation to use. Default is 'Affine'.
        See ANTsPy documentation for other options (https://antspy.readthedocs.io/en/latest/registration.html).

    Returns
    -------
    ANTsTransform
        The transformation object.
    """
    ref_img = ants.image_read(get_phantom_nii(phantom_weighting))
    reg = ants.registration(fixed=ref_img, moving=target_img, type_of_transform=xfm_type)
    return reg['fwdtransforms']

def warp_seg(target_img, xfm=None, weighting=None, seg='T1'):
    """Warp any segmentation to target image
    
    Parameters
    ----------
    target_img : ANTsImage
        The reference image.
    
    xfm : ANTsTransform
        The transformation object.

    weighting : str
        Which phantom weighting to use (T1 or T2).
    
    seg : str
        Which segmentation to use (T1, T2, ADC, LC, fiducials, wedges). Default is 'T1'.
    
    Returns
    -------
    ANTsImage
        The warped segmentation.
    """
    if xfm is None and weighting is None:
        raise ValueError('Either xfm or weighting must be provided')
    elif xfm is not None and weighting is not None:
        raise ValueError('xfm and weighting cannot both be provided')
    elif xfm is None and weighting is not None:
        xfm = reg_to_phantom(target_img, phantom_weighting=weighting, xfm_type='Affine')

    seg = ants.image_read(get_seg_nii(seg))
    seg_warp = ants.apply_transforms(fixed=target_img, moving=seg, whichtoinvert=[1], transformlist=xfm, interpolator='genericLabel')
    return seg_warp

def save_xfm(xfm, filename):
    """Save the transformation object to a file
    
    Parameters
    ----------
    xfm : ANTsTransform
        The transformation object.
    
    filename : str
        Filename of transform (file extension is ".mat" for affine transforms).
    """
    ants.write_transform(xfm, filename)

# def get_file_path_to_seg(seg='T1'):
#     """Get filename of segmentation image

#     Args:
#         seg (str, optional): Which segmentation (T1, T2, ADC, LC, fiducials, wedges). Default is 'T1'.

#     Raises:
#         ValueError: Wrong segmentation

#     Returns:
#         str: Full file path
#     """
#     avail_seg = ['T1', 'T2', 'ADC', 'LC', 'fiducials', 'wedges']
#     if seg not in avail_seg:
#         raise ValueError(f'Not a valid segmentation. (Valid: {avail_seg})')
#     else:
#         if seg == 'T1' or seg == 'T2' or seg == 'ADC':
#             return os.path.join(GHOSTDIR, 'data', f'{seg}_mimics.nii.gz')
#         elif seg == 'fiducials' or seg == 'wedges':
#             return os.path.join(GHOSTDIR, 'data', f'{seg}.nii.gz')
#         elif seg == 'LC':
#             return os.path.join(GHOSTDIR, 'data', f'{seg}_vials.nii.gz')


def process_all():
    # Shouldn' have any specific processing, just call other scripts
    pass