"""
Everything input and output of data
"""

import ants
import nibabel as nib
import numpy as np
import os
from bids.layout.models import BIDSImageFile
from nibabel.nifti1 import Nifti1Image
from numpy import ndarray

def load_4D_nifti(img, vol=None, mag=False):

    """
    Load 4D nifti image
    """
    nii = nib.load(img)
    if nii.header['datatype'] == 32:
        img_data = nii.get_fdata(dtype=np.complex64)
    else:
        img_data = nii.get_fdata()

    if mag:
        img_data = np.abs(img_data)
    
    if vol is not None:
        img_data = img_data[..., vol]
        nii.header['datatype'] = 64
    
    new_nii = nib.Nifti1Image(img_data, nii.affine, nii.header)
    ants_img = ants.from_nibabel(new_nii)

    return ants_img

def get_nifti_basename(fname):
    bname, ext = os.path.splitext(fname)
    if ext == '.gz':
        bname, ext = os.path.splitext(bname)
    return bname


def _get_image(x):    
    if type(x) == str:
        return ants.image_read(x)
    if type(x) == BIDSImageFile:
        return ants.image_read(x.path)
    if type(x) == Nifti1Image:
        return ants.from_nibabel(x)
    if type(x) == ndarray:
        return ants.from_numpy(x)
    else:
        return x