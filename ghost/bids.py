import os
import sys

def main_add_temperature():
    dicom_dir = sys.argv[1]
    nifti_dir = sys.argv[2]

    print(f'Dicom folder: {dicom_dir}')
    print(f"Nifti folder: {nifti_dir}")