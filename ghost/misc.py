from importlib import resources 
from ghost.phantom import *
import argparse
import ghost
import ants
import os

def generate_masks():
    parser = argparse.ArgumentParser(description='Generate T1, T2, and ADC masks from a given input file')
    parser.add_argument('input_file', help='input file path')
    parser.add_argument('ref', help='reference image for registration (T1 or T2)', type=str)
    parser.add_argument('--output_prefix', help='prefix for output files', type=str)
    parser.add_argument('--seg', default='all', help='segmentation image for registration (T1 or T2 or ADC)', type=str)
    args = parser.parse_args()

    input_file = args.input_file
    ref = args.ref
    output_prefix = args.output_prefix if args.output_prefix else os.path.splitext(os.path.basename(input_file))[0]

    if args.seg == 'all':
        seg = ['T1', 'T2', 'ADC']
    elif args.seg == 'T':
        seg = ['T1', 'T2']
    else:
        seg = [args.seg]

    # use input_file and output_prefix in your function code
    target_img = ants.image_read(input_file)

    # Find out if target_ANTsImage is T1w or T2w by looking at the metadata
    ref_img = ants.image_read(get_phantom_nii(ref))
    reg = ants.registration(fixed=ref_img, moving=target_img, type_of_transform='Affine')
    xfm = reg['fwdtransforms']

    for s in seg:
        seg_bin = ants.image_read(get_phantom_nii(s))
        warped_seg = ants.apply_transforms(fixed=target_img, moving=seg_bin, transformlist=xfm)
        # save each warped seg image to a file as output_prefix_seg.nii.gz
        
        # print "Created output_prefix_T1mask.nii.gz"
        print("Created " + output_prefix + '_' + s + 'mask.nii.gz')

def get_ghost_dir():
    p = os.path.join(resources.path(package=ghost, resource="").__enter__(), '..')
    return os.path.abspath(p)