import os
import json
import shlex
import shutil
import subprocess as sp
from datetime import datetime

import ants
import bids
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from skimage.draw import disk

from .metrics import calc_psnr
from .phantom import Caliber137

DERIVPATTERN = "sub-{subject}[/ses-{session}]/{tool}/sub-{subject}[_ses-{session}][_rec-{reconstruction}][_run-{run}][_desc-{desc}]_{suffix}.{extension}"

### Helper functions ###
def _logprint(s):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {s}")

def _update_layout(layout):
    return bids.BIDSLayout(root=layout.root, derivatives=layout.derivatives['derivatives'].root)

def _check_paths(fname):
    dir = os.path.dirname(fname)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    return fname

def _check_run(fname, ow):
    if (not os.path.exists(fname)) or ow:
        return True
    else:
        return False
    
def _make_fname(layout, ent):
    return layout.build_path(ent, DERIVPATTERN, validate=False)
    
def _make_deriv_fname(layout, ent, **kwargs):
    ent = ent.copy()
    for k in kwargs.keys():
        ent[k] = kwargs[k]
    return _check_paths(_make_fname(layout.derivatives['derivatives'], ent))

def _get_fname(layout, **kwargs):
    return layout.build_path(kwargs, DERIVPATTERN, validate=False)

def _get_seg_fname(layout, base_img, desc):
    ent = base_img.get_entities()
    try:
        run = ent['run']
    except KeyError:
        run = None

    return _get_fname(layout.derivatives['derivatives'], subject=ent['subject'], session=ent['session'], tool='ghost', 
                            reconstruction=ent['reconstruction'], run=run, desc=desc, suffix=ent['suffix'], extension=ent['extension'])

def _get_xfm_fname(layout, base_img, tool='ants', extension='mat', desc='0GenericAffine'):
    ent = base_img.get_entities()
    ent['tool'] = tool
    ent["desc"] = desc
    ent["extension"] = extension
    fname = _make_fname(layout.derivatives['derivatives'], ent)

    return fname

### Tools to use ###

def import_dicom_folder(dicom_dir, sub_name, ses_name, config, projdir):
    """
    Imports DICOM files from a specified directory into the BIDS format.

    Args:
        dicom_dir (str): The path to the directory containing the DICOM files.
        sub_name (str): The subject name for the BIDS dataset.
        ses_name (str): The session name for the BIDS dataset.
        config (str): The path to the configuration file for dcm2bids.
        projdir (str): The path to the project directory where the BIDS dataset will be created.

    Returns:
        None
    """

    cmd = f'dcm2bids -d {shlex.quote(dicom_dir)} -p {sub_name} -s {ses_name} -c {config} -o {projdir}/rawdata -l DEBUG'
    sp.Popen(shlex.split(cmd)).communicate()

def setup_bids_directories(projdir):
    """
    Set up the necessary BIDS directories and files for a project.

    Args:
        projdir (str): The path to the project directory.

    Returns:
        None
    """


    # Check for basic folders
    for f in ['rawdata', 'derivatives']:
        if not os.path.exists(f'{projdir}/{f}'):
            os.makedirs(f'{projdir}/{f}')
    
    # Check for dataset description
    if not os.path.exists(f'{projdir}/rawdata/dataset_description.json'):
        D = {"Name": "GHOST phantom rawdata", "BIDSVersion": "1.0.2"}
        with open(f'{projdir}/rawdata/dataset_description.json', 'w') as f:
            json.dump(D,f)
    
    if not os.path.exists(f'{projdir}/derivatives/dataset_description.json'):
        D = {"Name": "Ghost derivatives dataset", "BIDSVersion": "1.0.2", "GeneratedBy": "GHOST"}
        with open(f'{projdir}/derivatives/dataset_description.json', 'w') as f:
            json.dump(D,f)

def warp_mask(layout, bids_img, seg, phantom, weighting='T2w', ow=False):
    """
    Warps a segmentation mask to the space of a given image using ANTs registration.

    Parameters:
    layout (Layout): The BIDSLayout object representing the BIDS dataset.
    bids_img (BIDSImage): The BIDSImage object representing the input image.
    seg (str): The name of the segmentation mask to be warped.
    phantom (Phantom): The Phantom object representing the phantom used for registration.
    weighting (str, optional): The weighting scheme to be used during registration. Defaults to 'T2w'.
    ow (bool, optional): Flag indicating whether to overwrite existing results. Defaults to False.

    Returns:
    None
    """
    
    fname_aff = _get_xfm_fname(layout, bids_img, extension='.mat', desc='0GenericAffine')
    fname_syn = _get_xfm_fname(layout, bids_img, extension='.nii.gz', desc='1InverseWarp')
    xfm_calculated = False

    if ow:
        xfm_calculated = False
    elif os.path.exists(fname_syn) and os.path.exists(fname_aff):
        xfm = [fname_aff, fname_syn]
        xfm_calculated = True
        print("xfm already calculated")
    else:
        xfm = None
     
    fname_out = _make_deriv_fname(layout, bids_img.get_entities(), desc=f'seg{seg}', tool='ghost')

    if _check_run(fname_out, ow):
        seg_warp, xfm_out = phantom.warp_seg(ants.image_read(bids_img.path), seg, xfm, weighting)

        ants.image_write(seg_warp, _check_paths(fname_out))

        if not xfm_calculated:
            _check_paths(fname_aff)
            shutil.copy(xfm_out[0], fname_aff)
            shutil.copy(xfm_out[1], fname_syn)

def get_seg_loc(layout, bids_img, seg, phantom, offset=0):
    
    fname = _get_xfm_fname(layout, bids_img, extension='.mat', desc='0GenericAffine')
    tx = ants.read_transform(fname)
    z = phantom.get_phantom_location(seg) + offset
    p = [*tx.apply_to_point([0,0,z]), 1]
    affine = bids_img.get_image().affine
    ijk = np.linalg.inv(affine) @ p
    return ijk, p[:3]

def refine_mimics_2D_sag():
    return True

def refine_mimics_2D_cor():
    return True

def refine_mimics_2D(layout, bids_img, seg, phantom, ow=False):
    
    # Define the output
    ent = bids_img.get_entities()
    fname_2D = _make_deriv_fname(layout, ent, tool='ghost', desc=f'seg{seg}2D')

    if _check_run(fname_2D, ow):
        # Get slice with largest radius
        dx,dy,dz = ants.image_read(bids_img.path).spacing
        
        my_k = 0
        my_r = 0
        
        mimic_radius_mm = int(phantom.get_specs()['Sizes'][seg])/2
        mimic_radius_vox = mimic_radius_mm/dx

        ijk, xyz_ref = get_seg_loc(layout, bids_img, seg, phantom, offset=0)
        z0 = xyz_ref[2]
        print(ijk)
        print(xyz_ref)
        klist = [int(np.floor(ijk[2])), int(np.ceil(ijk[2]))]
        
        for k0 in klist:
            # Figure out which z position we are at relative to center of the sphere
            z1 = (bids_img.get_image().affine @ np.array([0,0,k0,1]))[2]
            
            if mimic_radius_vox**2 > (z0 - z1)**2:
                pv_rad = np.sqrt(mimic_radius_vox**2 - (z0 - z1)**2)
            else:
                pv_rad = 0
            
            if pv_rad > my_r:
                my_k = k0
                my_r = pv_rad

        seg_img = ants.image_read(_get_seg_fname(layout, bids_img, desc=f'seg{seg}'))
        
        new_seg = np.zeros(seg_img.shape[0:2])
        for i in range(1,15):
            com = center_of_mass(seg_img.numpy()==i)
            d = disk([com[0], com[1]], my_r, shape=seg_img.shape[0:2])
            new_seg[d] = i

        refined_seg = np.zeros(seg_img.shape)
        refined_seg[...,my_k] = new_seg
        refined_seg_img = ants.from_numpy(refined_seg, origin=seg_img.origin, direction=seg_img.direction, spacing=seg_img.spacing)

        # Make new filenames
        ants.image_write(refined_seg_img, fname_2D)

        return refined_seg_img

    else:
        return ants.image_read(fname_2D)

def find_best_slice(layout, bids_img, seg, slthick=5):
    z = []
    for i in [-1,0,1]:
        z.append*get_seg_loc(layout, bids_img, seg)
    
def get_fiducials(layout, bids_img, phantom, resample_res=[1.0, 1.0, 1.0], ow=False):
    """
    Get fiducials segmentation from an input image.

    Parameters:
    - layout (object): The BIDSLayout object representing the BIDS dataset.
    - bids_img (object): The BIDSImageFile object representing the input image.
    - phantom (object): The Phantom object representing the phantom used for fiducial segmentation.
    - resample_res (list, optional): The resolution to resample the input image to. Defaults to [1.0, 1.0, 1.0].
    - ow (bool, optional): Flag indicating whether to overwrite existing results. Defaults to False.

    Returns:
    - None

    This function performs the following steps:
    1. Interpolates swoop data if necessary.
    2. Checks for registration to template and creates a new transformation matrix if necessary.
    3. Runs phantom.fiducial_segmentation on the input image.
    4. Writes the fiducials and fiducial labels as output images.
    5. Saves the new transformation matrices to the BIDs structure.
    """

    ent = bids_img.get_entities()
    ent["reconstruction"] = ent["reconstruction"]+'Interp'
    fid_fname_out = _make_deriv_fname(layout, ent, tool='ghost', desc='segRegFid')

    if _check_run(fid_fname_out, ow):

        # Interpolate swoop data
        interp_fname = _make_deriv_fname(layout, ent, tool='ghost')
        
        if os.path.exists(interp_fname) and (not ow):
            swoop_img = ants.image_read(interp_fname)
        else:
            print(f"Resampling input image to {resample_res}")
            swoop_img = ants.resample_image(ants.image_read(bids_img.path), resample_params=resample_res, use_voxels=False, interp_type=4)
            ants.image_write(swoop_img, interp_fname)
        
        # Check for registration to template
        new_xfm = False
        xfm_fname = _make_deriv_fname(layout, ent, extension='mat', desc='Fiducials0GenericAffine', tool='ants')
        
        if os.path.exists(xfm_fname) and (not ow):
            xfm = xfm_fname
        else:
            new_xfm = True
            xfm = None

        # Run fiducial segmentation
        fiducials, fiducial_labels, xfm, refined_xfm = phantom.segment_fiducials(swoop_img, xfm=xfm, weighting='T2w', 
                                                                       binarize_threshold=0.5, verbose=True)
        
        # Write outputs
        ants.image_write(fiducials, _check_paths(fid_fname_out))

        fid_fname_out = _make_deriv_fname(layout, ent, tool='ghost', desc='segRegFidLabels')
        ants.image_write(fiducial_labels, _check_paths(fid_fname_out))


        if new_xfm:
            _check_paths(xfm_fname)
            shutil.copy(xfm, xfm_fname)

        for i,x in enumerate(refined_xfm):
            xfm_fname = _make_deriv_fname(layout, ent, extension='mat', desc=f'Fiducials0GenericAffine{i}', tool='ants')
            shutil.copy(x, xfm_fname)

def warp_thermo(layout, temp_bids_img, t2_bids_img, ow=False):
    
    out_fname = _make_deriv_fname(layout, temp_bids_img.get_entities(), tool='ghost', desc='regT2wN4')

    if _check_run(out_fname, ow):
        moving = ants.image_read(temp_bids_img.path)
        fixed = ants.image_read(t2_bids_img.path)
        reg = ants.registration(fixed, moving, type_of_transform='antsRegistrationSyN[s]')
        
        phantom_mask = ants.image_read(_make_deriv_fname(layout, t2_bids_img.get_entities(), desc=f'segphantomMask', tool='ghost'))
        N4 = ants.n4_bias_field_correction(ants.denoise_image(reg['warpedmovout'], mask=phantom_mask), mask=phantom_mask, return_bias_field=True)
        
        ants.image_write(reg['warpedmovout']/N4, out_fname)

def get_temperature(layout, thermo, phantom, plot_on=False):
    ent = thermo.get_entities()
    LC = layout.get(scope='derivatives', subject=ent['subject'], session=ent['session'], desc='segLC')[0].path

    if plot_on:    
        temperature, fig = phantom.loglike_temp(thermo, LC, plot_on)
        fig.show()
    else:
        temperature = phantom.loglike_temp(thermo, LC, plot_on)
    
    fname = _make_deriv_fname(layout, ent, extension='.txt', tool='stats', desc='temperature')
    with open(fname, 'w') as f:
        f.write(str(temperature))    

    return temperature

def calc_runs_psnr(layout, bids_img, ow=False):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two runs of an image.

    Parameters:
    layout (Layout): The BIDSLayout object representing the BIDS dataset.
    bids_img (BIDSImage): The BIDSImage object representing the image.
    ow (bool, optional): If True, overwrite existing PSNR file. Default is False.

    Returns:
    float: The calculated PSNR value.

    Raises:
    None

    """

    ent = bids_img.get_entities()
    fname = _make_deriv_fname(layout, ent, extension='.txt', tool='stats', desc='PSNR')

    if _check_run(fname, ow):

        img1 = layout.get(scope='raw', extension='.nii.gz', 
                    subject=ent['subject'], reconstruction=ent['reconstruction'], 
                    session=ent['session'], run=1)[0].path
        
        img2 = layout.get(scope='raw', extension='.nii.gz', 
                    subject=ent['subject'], reconstruction=ent['reconstruction'], 
                    session=ent['session'], run=2)[0].path
        
        phmask = layout.get(scope='derivatives', extension='.nii.gz', 
                    subject=ent['subject'], reconstruction=ent['reconstruction'], 
                    session=ent['session'], run=1, desc='segphantomMask')[0].path

        PSNR = calc_psnr(ants.image_read(img1), ants.image_read(img2), ants.image_read(phmask))
        with open(fname, 'w') as f:
            f.write(str(PSNR))

        return PSNR

def parse_fiducial_positions(layout, img, phantom, ow=False):
    """
    Parse fiducial positions from an image and calculate the differences between the parsed positions and reference positions.

    Parameters:
    - layout (Layout): The BIDSLayout object representing the layout of the dataset.
    - img (Image): The image object from which to parse the fiducial positions.
    - phantom (Phantom): The phantom object containing the reference fiducial locations.
    - ow (bool): Flag indicating whether to overwrite existing files. Default is False.

    Returns:
    - positions (ndarray): A numpy array containing the parsed fiducial positions.

    """

    ent = img.get_entities()
    
    try:
        run = ent['run']
    except KeyError:
        run = None
    
    fname = _make_deriv_fname(layout, ent, extension='.csv', tool='stats', desc='FidPos')
    
    if _check_run(fname, ow):
        seg = layout.get(scope='derivatives', suffix='T2w', subject=ent['subject'], 
                        session=ent['session'], desc='segRegFid', run=run, reconstruction=ent['reconstruction']+"Interp")[0]

        fiducials = ants.image_read(seg.path)
        ent = seg.get_entities()

        affine = np.array(phantom.get_specs()['FiducialAffine'])

        # True space positions
        positions = np.zeros([3,15])
        for i in range(15):
            p = ants.get_center_of_mass(ants.slice_image(fiducials, axis=3, idx=i))

            # The affine we get is from 3T to Swoop space
            xfm = ants.read_transform(layout.get(scope='derivatives', suffix='T2w', run=run, subject=ent['subject'], 
                                                reconstruction=ent['reconstruction'], session=ent['session'], desc=f'Fiducials0GenericAffine{i}')[0])
            p_3T = np.array([*xfm.apply_to_point(p),1])
            p_ref = (affine @ p_3T)[:3]
            positions[:,i] = p_ref

        ref_pos = phantom.get_ref_fiducial_locations()
        label = np.arange(1,16)
        df = pd.DataFrame(label, columns=['label'])
        df['X'] = positions[0,:]
        df['Y'] = positions[1,:]
        df['Z'] = positions[2,:]
        df['refX'] = ref_pos[0,:]
        df['refY'] = ref_pos[1,:]
        df['refZ'] = ref_pos[2,:]
        df['diffX'] = df['X']-df['refX']
        df['diffY'] = df['Y']-df['refY']
        df['diffZ'] = df['Z']-df['refZ']
        
        fname = _make_deriv_fname(layout, ent, extension='.csv', tool='stats', desc='FidPos')
        df.to_csv(fname)

        return positions

def get_intensity_stats(layout, bids_img, seg_name, ow=False):
    """
    Calculate intensity statistics for a given image and segmentation.

    Parameters:
    - layout (Layout): The BIDSLayout object representing the BIDS dataset.
    - bids_img (BIDSImage): The BIDSImage object representing the input image.
    - seg_name (str): The name of the segmentation.
    - ow (bool): Flag indicating whether to overwrite existing files (default: False).

    Returns:
    None
    """

    ent = bids_img.get_entities()
    fname = _make_deriv_fname(layout, ent, extension='.csv', tool='stats', desc=seg_name)
    
    try:
        run = ent['run']
    except KeyError:
        run = None
    
    if _check_run(fname, ow):
        seg = layout.get(scope='derivatives', suffix='T2w', subject=ent['subject'], session=ent['session'],
                        run=run, reconstruction=ent['reconstruction'], desc=seg_name)[0]
        
        df = ants.label_stats(ants.image_read(bids_img.path), ants.image_read(seg.path))
        df = df.drop(df[df['LabelValue'] == 0.0].index)
        df.drop(['t', 'Count', 'Mass'], axis=1, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(fname)
 
def unity_qa_process_subject(layout, sub, ses):
    """
    Process the Unity QA data for a subject.

    Args:
        layout (Layout): The BIDS Layout object.
        sub (str): The subject ID.
        ses (str): The session ID.

    Returns:
        None
    """
    
    phantom = Caliber137()
    axi1 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=1)[0]
    axi2 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=2)[0]
    sag = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='sag', session=ses)[0]
    cor = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='cor', session=ses)[0]
    fisp = layout.get(scope='raw', extension='.nii.gz', subject=sub, suffix='PDw', session=ses)[0]

    _logprint(f'Starting for {sub}-{ses}')
    
    # Warp the masks
    _logprint("--- Warping masks ---")
    for img in [axi1, axi2, sag, cor]:
        _logprint(img.filename)
        for mask in ['T1mimics', 'T2mimics', 'ADC', 'LC', 'phantomMask', 'SNR']:
            _logprint(mask)
            warp_mask(layout, img, mask, phantom, ow=False)
            
    layout = _update_layout(layout)

    # T1, T2, ADC arrays
    _logprint("Refining masks to 2D and getting intensity stats")
    for img in [axi1, axi2, sag, cor]:
        _logprint(img.filename)
        for mask in ['T1mimics', 'T2mimics', 'ADC']:
            refine_mimics_2D(layout, img, mask, phantom, ow=False)
            
            layout = _update_layout(layout)
            get_intensity_stats(layout, img, f"seg{mask}", ow=False)
    
    layout = _update_layout(layout)
    
    # Fiducials
    _logprint("Getting fiducial arrays")
    for img in [axi1, axi2, sag, cor]:
        _logprint(img.filename)
        get_fiducials(layout, img, phantom, ow=False)

    layout = _update_layout(layout)

    # Fiducial positions
    _logprint("Parsing fiducial locations")
    for img in [axi1, axi2, sag, cor]:
        _logprint(img.filename)
        parse_fiducial_positions(layout, img, phantom, ow=False)

    # PSNR
    _logprint("Calculating PSNR")
    calc_runs_psnr(layout, axi1, ow=False)

    # Tempearture
    _logprint("Getting temperature")
    warp_thermo(layout, fisp, axi1, ow=False)
    layout = _update_layout(layout)

    thermo = layout.get(scope='derivatives', suffix='PDw', subject=sub, session=ses, desc='regT2wN4')[0]
    get_temperature(layout, thermo, phantom)
