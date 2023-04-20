"""
Everything input and output of data
"""

import ants
import nibabel as nib
import numpy as np

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