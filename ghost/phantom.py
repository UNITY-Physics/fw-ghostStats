from . import GHOSTDIR
import subprocess as sp
import os
import pandas as pd

"""
Functions to deal with the phantom images
"""

def get_phantom_nii(weighting='T1'):
    """Get filename of phantom image

    Args:
        weighting (str, optional): Which weighting (T1 or T2). Defaults to 'T1'.

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
        seg (str, optional): Which segmentation (T1, T2, ADC, LC). Defaults to 'T1'.

    Raises:
        ValueError: Wrong segmentation

    Returns:
        str: Full file path
    """
    avail_seg = ['T1', 'T2', 'ADC', 'LC']
    if seg not in avail_seg:
        raise ValueError(f'Not a valid segmentation. (Valid: {avail_seg})')
    else:
        return os.path.join(GHOSTDIR, 'data', f'{seg}_vials.nii.gz')

def download_ref_data():
    """
    Downloads reference data for the phantom from Dropbox
    """
    files = [{'fname':'T1_phantom.nii.gz',
            'link':'https://www.dropbox.com/s/cwujos81rtt6s87/T1_phantom.nii.gz?dl=0'},
            {'fname':'T2_phantom.nii.gz',
            'link':'https://www.dropbox.com/s/pcq7be6q019j6jb/T2_phantom.nii.gz?dl=0'},
            {'fname':'ADC_vials.nii.gz',
            'link':'https://www.dropbox.com/s/yuf0sl9uz1bkqu5/ADC_vials.nii.gz?dl=0'},
            {'fname':'LC_vials.nii.gz',
            'link':'https://www.dropbox.com/s/1c2mjugtyb04xjp/LC_vials.nii.gz?dl=0'},
            {'fname':'T2_vials.nii.gz',
            'link':'https://www.dropbox.com/s/vkkwd02f8dz2nqu/T2_vials.nii.gz?dl=0'},
            {'fname':'T1_vials.nii.gz',
            'link':'https://www.dropbox.com/s/0ai6z3cg94xcvn8/T1_vials.nii.gz?dl=0'}
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

def reg_to_phantom():
    pass

def process_all():
    # Shouldn' have any specific processing, just call other scripts
    pass