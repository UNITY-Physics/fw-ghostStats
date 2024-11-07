import json
import os
import shlex
import shutil
import subprocess as sp
from datetime import datetime

import ants
import bids
import numpy as np
import pandas as pd
from skimage.metrics import normalized_mutual_information

from .phantom import Caliber137
from .utils import calc_psnr, calc_ssim
from .ml import run_prediction

DERIVPATTERN = "sub-{subject}[/ses-{session}]/{tool}/sub-{subject}[_ses-{session}][_rec-{reconstruction}][_run-{run}][_desc-{desc}]_{suffix}.{extension}"
nnUNet_config = '/home/em2876lj/Code/GHOST/nnUnet_models/models.json'
### Helper functions ###

def _logprint(s):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {s}", flush=True)


def copy_file(fsource, fdest):
    shutil.copy(fsource, _check_paths(fdest))


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

def import_dicom_folder(dicom_dir, sub_name, ses_name, config, projdir, skip_dcm2niix=False):
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

    cmd = f'dcm2bids --force_dcm2bids -d {shlex.quote(dicom_dir)} -p {sub_name} -s {ses_name} -c {config} -o {projdir}/rawdata -l DEBUG'
    if skip_dcm2niix:
        cmd += ' --skip_dcm2niix'
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
    
    def dump_description(fname, D):
        if not os.path.exists(fname):
            with open(fname, 'w') as f:
                json.dump(D,f,indent=4)

    # Check for dataset description    
    dump_description(fname=f'{projdir}/rawdata/dataset_description.json',
                     D={"Name": "GHOST phantom rawdata", "BIDSVersion": "1.0.2"})
    
    dump_description(fname=f'{projdir}/derivatives/dataset_description.json',
                     D={"Name": "Ghost derivatives dataset", "BIDSVersion": "1.0.2", "GeneratedBy": [{"Name":"GHOST"}]})

    dump_description(fname = f'{projdir}/dataset_description.json',
                     D={"Name": "UNITY QA example dataset", "BIDSVersion": "1.0.2"})
    

def warp_mask(layout, bids_img, seg, phantom, xfm_type='SyN', weighting='T2w', ow=False):
        
    fname_out = _make_deriv_fname(layout, bids_img.get_entities(), desc=f'seg{seg}', tool='ghost')

    if xfm_type.lower() == 'syn':
        xfm = [_get_xfm_fname(layout, bids_img, extension='.mat', desc='0GenericAffine'),
               _get_xfm_fname(layout, bids_img, extension='.nii.gz', desc='1InverseWarp')]
    else:
        xfm = [_get_xfm_fname(layout, bids_img, extension='.mat', desc='0GenericAffine')]

    if _check_run(fname_out, ow):
        seg_warp = phantom.warp_seg(target_img=ants.image_read(bids_img.path), xfm=xfm, seg=seg)
        ants.image_write(seg_warp, _check_paths(fname_out))
    
    return fname_out

def reg_img(layout, bids_img, phantom, do_syn=False, weighting='T2w', ow=False):
    
    _logprint(f'Calculating transformation to template')
    
    fname_aff = _get_xfm_fname(layout, bids_img, extension='.mat', desc='0GenericAffine')
    fname_InvSyn = _get_xfm_fname(layout, bids_img, extension='.nii.gz', desc='1InverseWarp')
    fname_FwdSyn = _get_xfm_fname(layout, bids_img, extension='.nii.gz', desc='1Warp')

    if _check_run(fname_aff, ow):

        inv_xfm, fwd_xfm = phantom.reg_to_phantom(ants.image_read(bids_img.path), do_syn=do_syn, weighting=weighting)
        
        _logprint(f'Done. {inv_xfm}')
        copy_file(inv_xfm[0], fname_aff)
        
        if len(inv_xfm) > 1:
            
            copy_file(inv_xfm[1], fname_InvSyn)
            copy_file(fwd_xfm[0], fname_FwdSyn)
    
    else:
        print("xfm already calculated")
        

def get_seg_loc(layout, bids_img, seg, phantom, offset=0):
    
    fname = _get_xfm_fname(layout, bids_img, extension='.mat', desc='0GenericAffine')
    tx = ants.read_transform(fname)
    z = phantom.get_phantom_location(seg) + offset
    p = [*tx.apply_to_point([0,0,z]), 1]
    affine = bids_img.get_image().affine
    ijk = np.linalg.inv(affine) @ p
    return ijk, p[:3]


def refine_mimics_2D_axi(layout, bids_img, seg, phantom, ow=False):
    
    # Define the output
    ent = bids_img.get_entities()
    fname_2D = _make_deriv_fname(layout, ent, tool='ghost', desc=f'seg{seg}2D')

    if _check_run(fname_2D, ow):
        seg_img = ants.image_read(_get_seg_fname(layout, bids_img, desc=f'seg{seg}'))
        xfm = _get_xfm_fname(layout, bids_img, extension='.mat', desc='0GenericAffine')

        refined_seg_img = phantom.mimic_3D_to_2D_axial(seg_img=seg_img, seg_name=seg, xfm_fname=xfm, radius=None)
        ants.image_write(refined_seg_img, fname_2D)

    return fname_2D


def find_best_slice(layout, bids_img, seg, slthick=5):
    z = []
    for i in [-1,0,1]:
        z.append*get_seg_loc(layout, bids_img, seg)

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


def segment_fiducials(layout, img, device='cpu', ow=False):
    
    ent = img.get_entities()
    fname = _make_deriv_fname(layout, ent, extension='.nii.gz', tool='ghost', desc='segFidLabelsUNetAxis')
    
    if _check_run(fname, ow):
        run_prediction(input=img.path, output=fname, scan_plane=ent['reconstruction'], device=device, keep=False)

    return fname

### Stats ###
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
    fname = _make_deriv_fname(layout, ent, extension='.csv', tool='stats', desc='PSNR')


    if _check_run(fname, ow):

        img1 = layout.get(scope='raw', extension='.nii.gz', 
                    subject=ent['subject'], reconstruction=ent['reconstruction'], 
                    session=ent['session'], run=1)[0].get_image().get_fdata()
        
        img2 = layout.get(scope='raw', extension='.nii.gz', 
                    subject=ent['subject'], reconstruction=ent['reconstruction'], 
                    session=ent['session'], run=2)[0].get_image().get_fdata()
        
        phmask = layout.get(scope='derivatives', extension='.nii.gz', 
                    subject=ent['subject'], reconstruction=ent['reconstruction'], 
                    session=ent['session'], run=1, desc='segphantomMask')[0].get_image().get_fdata()
        
        # Normalize images to range 0-1
        img1 /= np.quantile(img1[...], 0.99)
        img2 /= np.quantile(img2[...], 0.99)

        MSE, PSNR = calc_psnr(img1, img2, phmask)
        NMI = normalized_mutual_information(img1*phmask, img2*phmask)
        SSIM = calc_ssim(img1, img2, phmask)

        df = pd.DataFrame({"MSE":[MSE], "PSNR":[PSNR], "NMI":[NMI], "SSIM":[SSIM]})
        df.to_csv(fname)
        _logprint(f"Wrote SNR file to {fname}")

    return fname, PSNR

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
                        session=ent['session'], desc='segRegFid', run=run, reconstruction=ent['reconstruction'])[0]

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
        
        df.to_csv(fname)

        return positions

def get_fiducial_positions2(layout, img, phantom, out_stat='FidPosReal', input_desc='segRegFidLabels', aff_fname='FidPointAffine', transform_type='affine', ow=False):

    ent = img.get_entities()
    
    try:
        run = ent['run']
    except KeyError:
        run = None
    
    fname = _make_deriv_fname(layout, ent, extension='.csv', tool='stats', desc=out_stat)
    
    if _check_run(fname, ow):
        seg_fname = layout.get(scope='derivatives', suffix='T2w', subject=ent['subject'], 
                        session=ent['session'], desc=input_desc, run=run, reconstruction=ent['reconstruction'])[0]
        
        seg = ants.image_read(seg_fname.path)

        # Label stats
        seg_df = ants.label_stats(image=seg, label_image=seg)
        seg_df = seg_df[seg_df.LabelValue > 0]
        seg_df.drop(columns=['Mean', 'Min', 'Max', 'Variance', 'Count', 'Volume', 'Mass', 't'], inplace=True)
        seg_df.reset_index(inplace=True, drop=True)

        # Get positions
        seg_pos = np.array([seg_df.x, seg_df.y, seg_df.z]).T
        ref_loc = phantom.get_ref_fiducial_locations().T

        # Affine registration with points
        reg = ants.fit_transform_to_paired_points(seg_pos, ref_loc, transform_type=transform_type)
        new_points = np.zeros_like(ref_loc)
        for i in range(ref_loc.shape[0]):
            new_points[i,:] = ants.apply_ants_transform_to_point(reg.invert(), seg_pos[i,:])

        # Create new dataframe with the results
        seg_df.rename(columns={'x':'x_org','y':'y_org','z':'z_org'}, inplace=True)
        seg_df['x_ref'] = ref_loc[:,0]
        seg_df['y_ref'] = ref_loc[:,1]
        seg_df['z_ref'] = ref_loc[:,2]

        seg_df['x_reg'] = new_points[:,0]
        seg_df['y_reg'] = new_points[:,1]
        seg_df['z_reg'] = new_points[:,2]

        seg_df['x_diff'] = seg_df['x_ref'] - seg_df['x_reg']
        seg_df['y_diff'] = seg_df['y_ref'] - seg_df['y_reg']
        seg_df['z_diff'] = seg_df['z_ref'] - seg_df['z_reg']

        seg_df.to_csv(fname)

        # Save reg file
        fname = _make_deriv_fname(layout, ent, extension='.txt', tool='ants', desc=aff_fname)
        mat = reg.parameters.reshape([4,3])
        np.savetxt(fname, mat)

        return seg_df, fname

def get_fiducial_points2(layout, img, phantom, ow=False):
    # 1. Find position of fiducial in template space
    fid = ants.image_read(phantom.get_seg_nii('fiducials'))
    temp_points = np.zeros((15,3))

    for i in range(15):
        temp_points[i,:] = ants.get_center_of_mass(fid==(i+1))

    # 2. Get position in swoop space using affine xfm
    # dp = phantom.get_specs()['FiducialPositions']
    # design_points = np.zeros((15,3))
    swoop_points = np.zeros((15,3))
    reg_points = np.zeros((15,3))

    ent = img.get_entities()

    for i in range(15):
        fname = layout.get(scope='derivatives', suffix='T2w', run=1, subject=ent['subject'], 
                        reconstruction=ent['reconstruction'], session=ent['session'], desc=f'Fiducials0GenericAffine{i}')[0]
        xfm = ants.read_transform(fname)
        swoop_points[i,:] = ants.apply_ants_transform_to_point(xfm.invert(), temp_points[i,:])
        
        # design_points[i,:] = dp[f'{i+1}']
        
    reg = ants.fit_transform_to_paired_points(swoop_points, temp_points, transform_type='rigid')
    
    for i in range(15):
        reg_points[i,:] = reg.invert().apply_to_point(swoop_points[i,:])

    return reg_points, temp_points

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
    
    return fname

def get_fiducial_position_nnuNet(layout, img, phantom, out_stat='FidPosUNetAxis', input_desc='segFidLabelsUNetAxis', aff_fname='FidPointAffine', ow=False):
    
    ent = img.get_entities()
    
    try:
        run = ent['run']
    except KeyError:
        run = None
    
    fname = _make_deriv_fname(layout, ent, extension='.csv', tool='stats', desc=out_stat)
    
    if _check_run(fname, ow):
    
        seg_fname = layout.get(scope='derivatives', suffix='T2w', subject=ent['subject'], 
                        session=ent['session'], desc=input_desc, run=run, reconstruction=ent['reconstruction'])[0]
        
        seg = ants.image_read(seg_fname.path)

        all_reg, dfs = phantom.point_reg_fiducials_2D(seg, acq_axis=ent['reconstruction'])

        mat = np.zeros((len(all_reg),6))
        for i in range(len(all_reg)):
            mat[i,:] = all_reg[i]

        np.savetxt(_make_deriv_fname(layout, ent, extension='.txt', tool='ants', desc=aff_fname), mat)
        
        seg_df = pd.concat(dfs)
        seg_df.to_csv(fname)


### Study specific workflows ###
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
    axi = []
    sag = None
    cor = None
    fisp = None
    
    # Axial
    files_axi = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses)
    if len(files_axi)==2:
        try:
            axi1 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=1)[0]
            axi2 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=2)[0]
            axi = [axi1, axi2]
            print(f'Found {len(files_axi)} axial scans')
        except:
            print("Expected two axial images with run 1 and 2. But did not work")
    
    elif len(files_axi)==1:
        axi.append(layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses)[0])
        print(f'Found {len(files_axi)} axial scans')

    else:
        print(f'Expected to find at least 1 axial scan. Found {len(files_axi)} axial scans')

    # Sag
    files_sag = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='sag', session=ses)
    if len(files_sag) == 1:
        sag = files_sag[0]
    else:
        print(f"Expected to find only one sag scan. Found {len(files_sag)}")

    # Cor
    files_cor = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='cor', session=ses)
    if len(files_cor) == 1:
        cor = files_cor[0]
    else:
        print(f"Expected to find 1 cor scan. Found {len(files_cor)}")

    # FISP
    files_fisp = layout.get(scope='raw', extension='.nii.gz', subject=sub, suffix='PDw', session=ses)
    if len(files_fisp) == 1:
        fisp = files_fisp[0]
    else:
        print(f"Expected to find 1 fisp scan. Found {len(files_cor)}")

    _logprint(f'Starting for {sub}-{ses}')
    
    mimics = ['T1mimics', 'T2mimics', 'ADC']
    all_masks = [*mimics, 'LC', 'phantomMask']
    all_t2 = [*axi, sag, cor]

    # Warp the masks
    _logprint("--- Register to template ---")
    for img in all_t2:
        reg_img(layout, img, phantom, ow=False)
    
    _logprint('Warping masks')
    layout = _update_layout(layout)
    for img in all_t2:
        if img:
            for mask in all_masks:
                warp_mask(layout, img, mask, phantom, xfm_type='Rigid', ow=False)

    _logprint('Refining 2D masks')
    layout = _update_layout(layout)
    for img in axi:
        for mask in mimics:
            refine_mimics_2D_axi(layout, img, mask, phantom, ow=False)

    _logprint('Getting intensity stats')            
    layout = _update_layout(layout)
    for img in axi:
        for mask in mimics:
            get_intensity_stats(layout, img, f"seg{mask}", ow=False)
    
    # _logprint("Getting fiducial arrays")
    # layout = _update_layout(layout)
    # for img in all_t2:
        # get_fiducials(layout, img, phantom, ow=False)
    
    # _logprint("Parsing fiducial locations")
    # for img in [axi1, axi2, sag, cor]:
        # _logprint(img.filename)
        # get_fiducial_positions2(layout, img, phantom, out_stat='FidPosUNet', input_desc='segFidLabelsUNet', aff_fname='FidPointUNetAffine', ow=True)
    
    _logprint("Calculating PSNR")
    if len(axi)==2:
        calc_runs_psnr(layout, axi[0], ow=True)

