#!/usr/bin/env python
"""The GHOST Run Script"""
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime
import bids

from flywheel_gear_toolkit import GearToolkitContext

import ghost.bids as gb
from ghost.phantom import Caliber137
from utils.parser import parse_config, download_dataset


log = logging.getLogger(__name__)

def main(context: GearToolkitContext) -> None:

    try:
        my_id = sys.argv[1]
        gb._logprint(f"Running with user supplied ID: {my_id}")
    except:
        gb._logprint("No ID supplied. Reading from input file")
        my_id = None
        
    container, config, manifest, inputs = parse_config(context, input_id=my_id)
    
    subses = download_dataset(context, container, config)

    gb._logprint("Indexing folder structure")    
    layout = bids.BIDSLayout(root=f'{config["work_dir"]}/rawdata', derivatives=f'{config["work_dir"]}/derivatives')

    gb._logprint("Running main script...")
    
    for sub in subses.keys():
        for ses in subses[sub].keys():
            raw_fnames, deriv_fnames = fw_process_subject(layout, sub, ses, 
                                                  run_mimics=config["runMimicSeg"], 
                                                  run_fiducials=config["runFiducialSeg"],
                                                  unet_device=config["nnUNetDevice"],
                                                  unet_quick=config['nnUNetQuick'])
    
            out_files = []
            out_files.extend(raw_fnames)
            out_files.extend(deriv_fnames)

            # Create a new analysis
            gversion = manifest["version"]
            gname = manifest["name"]
            gdate = datetime.now().strftime("%Y%M%d_%H:%M:%S")
            image = manifest["custom"]["gear-builder"]["image"]
            session_container = context.client.get(subses[sub][ses])
            
            analysis = session_container.add_analysis(label=f'{gname}/{gversion}/{gdate}')
            analysis.update_info({"gear":gname,
                                  "version":gversion, 
                                  "image":image,
                                  "Date":gdate,
                                  **config})


            for file in out_files:
                gb._logprint(f"Uploading output file: {os.path.basename(file)}")
                analysis.upload_output(file)

    gb._logprint("Copying output files")

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])



def parse_input_files(layout, sub, ses, show_summary=True):

    my_files = {'axi':[], 'sag':[], 'cor':[]}

    for ax in my_files.keys():
        files = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction=ax, session=ses)
        
        if ax == 'axi':

            if len(files)==2:
                axi1 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=1)[0]
                axi2 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=2)[0]
                my_files['axi'] = [axi1, axi2]

            elif len(files)==1:
                my_files['axi'] = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses)
            
            else:
                warnings.warn(f'Expected to find 1 or 2 axial scans. Found {len(files)} axial scans')

        else:
            if len(files) == 1:
                my_files[ax] = files
            elif len(files) > 1:
                my_files[ax] = [files[0]]
            else:
                warnings.warn(f"Found no {ax} scans")
    
    if show_summary:
        print(f"--- SUB: {sub}, SES: {ses} ---")
        print(f"Axial: {len(my_files['axi'])} scans")
        print(f"Cor: {len(my_files['cor'])} scans")
        print(f"Sag: {len(my_files['sag'])} scans")

    return my_files


def fw_process_subject(layout, sub, ses, run_mimics=True, run_fiducials=True, unet_device='cpu', unet_quick=False):
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
    my_files = parse_input_files(layout, sub, ses)
    print(my_files)
    
    gb._logprint(f'Starting for {sub}-{ses}')
    
    mimics = ['T1mimics', 'T2mimics', 'ADC']
    all_masks = [*mimics, 'phantomMask']
    all_t2 = [*my_files['axi'], *my_files['sag'], *my_files['cor']]

    deriv_fnames = []
    raw_fnames = [x.path for x in all_t2]

    gb._logprint("--- Register to template ---")
    for img in all_t2:
        gb.reg_img(layout, img, phantom, ow=False)

    if run_mimics:
        gb._logprint('Warping masks')
        layout = gb._update_layout(layout)
        for img in all_t2:
            for mask in all_masks:
                try:
                    fname = gb.warp_mask(layout, img, mask, phantom, xfm_type='Rigid', ow=False)
                    deriv_fnames.append(fname)
                except:
                    warnings.warn(f"Failed to run warp_mask for mask {mask} and {img}")

        gb._logprint('Refining 2D masks')
        layout = gb._update_layout(layout)
        for img in my_files['axi']:
            for mask in mimics:
                try:
                    fname = gb.refine_mimics_2D_axi(layout, img, mask, phantom, ow=False)
                    deriv_fnames.append(fname)
                except:
                    warnings.warn(f"Failed to run refine_mimics_2D_axi for mask {mask} and {img}")

        gb._logprint('Getting intensity stats')            
        layout = gb._update_layout(layout)
        for img in my_files['axi']:
            for mask in mimics:
                try:
                    fname = gb.get_intensity_stats(layout, img, f"seg{mask}", ow=False)
                    deriv_fnames.append(fname)
                except:
                    warnings.warn(f"Failed to run get_intensity_stats for mask {mask} and {img}")

    if run_fiducials:
        gb._logprint("Segmenting fiducial arrays")
        layout = gb._update_layout(layout)
        for img in my_files['axi']:
            try:
                fname = gb.segment_fiducials(layout, img, device=unet_device, quick=unet_quick, ow=False)
                deriv_fnames.append(fname)
            except:
                warnings.warn(f"Failed to run segment_fiducials on device={unet_device} for image {img}")

    return raw_fnames, deriv_fnames

if __name__ == "__main__":  # pragma: no cover
    
    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        main(gear_context)