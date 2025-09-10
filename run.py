#!/usr/bin/env python
"""The GHOST Stats Run Script"""
import fnmatch
import json
import logging
import os
import re
import sys

from datetime import datetime

import bids
from flywheel_gear_toolkit import GearToolkitContext

import ghost.bids as gb
from ghost.phantom import Caliber137

log = logging.getLogger(__name__)

def main(context: GearToolkitContext) -> None:
    
    try:
        my_id = sys.argv[1]
        gb._logprint(f"Running with user supplied ID: {my_id}")
    except:
        gb._logprint("No ID supplied. Reading from context container")
        my_id = None

    container, config, manifest, inputs = parse_config(context, input_id=my_id)

    os.makedirs(os.path.join(config['work_dir']), exist_ok=True)
    gb.setup_bids_directories(config['work_dir'])

    # Parse all the data in the container
    dataset_ids = get_dataset_ids(context, container)
    for sub in dataset_ids.keys():
        for ses in dataset_ids[sub]['sessions'].keys():
            gb._logprint(f"Parsing data for {sub} {ses}")
            container = context.client.get(dataset_ids[sub]['sessions'][ses])
            download_session(container, config)

    # Initialize the BIDS layout
    layout = bids.BIDSLayout(root=config['work_dir'], derivatives=os.path.join(config['work_dir'],'derivatives'), validate=False)
    
    # Run processing for each session
    phantom = Caliber137()
    for sub in dataset_ids.keys():
        for ses in dataset_ids[sub]['sessions'].keys():
            gb._logprint(f"Processing data for {sub} {ses}")
            try:
                process_session(layout, phantom, sub, ses)
            except:
                gb._logprint(f"Could not process {sub} {ses}")

    # Collect output files, first reinitialize the bids layout with all the new files
    layout = bids.BIDSLayout(root=config['work_dir'], derivatives=os.path.join(config['work_dir'],'derivatives'), validate=False)
    gb._logprint("Copying output files")

    if not os.path.exists(config['output_dir']): # Not made when running locally
        os.makedirs(config['output_dir'])
    
    for sub in dataset_ids.keys():
        
        for ses in dataset_ids[sub]['sessions'].keys():
            csv_files = [x.path for x in layout.get(scope='derivatives', subject=sub, session=ses, extension='.csv')]
            gb._logprint(f"Uloading data for {sub} {ses}")
            
            if len(csv_files) > 0:
                # Create a new analysis
                gversion = manifest["version"]
                gname = manifest["name"]
                gdate = datetime.now().strftime("%Y%M%d_%H:%M:%S")
                image = manifest["custom"]["gear-builder"]["image"]
                session_container = context.client.get(dataset_ids[sub]['sessions'][ses])
                
                analysis = session_container.add_analysis(label=f'{gname}/{gversion}/{gdate}')
                analysis.update_info({"gear":gname,
                                    "version":gversion, 
                                    "image":image,
                                    "Date":gdate,
                                    **config})

                for file in csv_files:
                    gb._logprint(f"Uploading output file: {os.path.basename(file)}")
                    analysis.upload_output(file)
            else:
                print(f"No output files for {sub} {ses}")



def process_session(layout, phantom, sub, ses):

    # Calculate PSNR
    try:
        axi1 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=1)[0]
        gb.calc_runs_psnr(layout, axi1)
    except:
        print("Failed to calculate SNR")

    # Calculate distortions
    try:
        axi1 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=1)[0]
        gb.get_fiducial_positions2(layout, axi1, phantom, out_stat='FidPosUNet', input_desc='segFidLabelsUNetAxis', aff_fname='FidPointUNetAffine', ow=True)
    except:
        print("Failed to calculate fiducial positions for run 1")

    try:
        axi2 = layout.get(scope='raw', extension='.nii.gz', subject=sub, reconstruction='axi', session=ses, run=2)[0]
        gb.get_fiducial_positions2(layout, axi2, phantom, out_stat='FidPosUNet', input_desc='segFidLabelsUNetAxis', aff_fname='FidPointUNetAffine', ow=True)
    except:
        print("Failed to calculate fiducial positions for run 2")

def parse_config(gear_context: GearToolkitContext, input_id: str):
    
    if not input_id:
        destination_id = gear_context.destination.get("id")
        try:
            container = gear_context.client.get(destination_id)
            input_id = container.parent.id

        except Exception as e:
            log.error(e, exc_info=True)
            sys.exit(1)

    base_dir = '/flywheel/v0'

    input_dir = base_dir + '/input/'
    work_dir = base_dir + '/work/'
    output_dir = base_dir + '/output/'

    container = gear_context.client.get(input_id)
    
    # Read config.json file
    with open(base_dir + '/config.json') as f:
        config = json.load(f)

    # Read manifest.json file
    with open(base_dir + '/manifest.json') as f:
        manifest = json.load(f)
    
    inputs = config['inputs']
    
    config = config['config']
    config['input_dir'] = input_dir
    config['work_dir'] = work_dir
    config['output_dir'] = output_dir

    return container, config, manifest, inputs


def download_session(container, config):
    
    my_analysis = find_latest_analysis(container.reload().analyses)
    
    if my_analysis:
        fname = my_analysis.files[0].name
        sub = re.findall(r'sub-(.*?)_', fname)[0]
        ses = re.findall(r'ses-(.*?)_', fname)[0]

        deriv_dir = os.path.join(config['work_dir'], 'derivatives', f'sub-{sub}', f'ses-{ses}')
        raw_dir = os.path.join(config['work_dir'], 'rawdata', f'sub-{sub}', f'ses-{ses}')

        os.makedirs(deriv_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(os.path.join(deriv_dir, 'ghost'), exist_ok=True)
        os.makedirs(os.path.join(deriv_dir, 'stats'), exist_ok=True)

        # Types of analysis we want to do:
        raw_files = []
        try:
            raw_files.append(find_file(my_analysis, reconstruction='axi', run=1, suffix='T2w', extension='.nii.gz')[0])
        except IndexError:
            gb._logprint(f'Could not find axial run 1')
        
        try:
            raw_files.append(find_file(my_analysis, reconstruction='axi', run=2, suffix='T2w', extension='.nii.gz')[0])
        except IndexError:
            gb._logprint(f'Could not find axial run 2')

        for f in raw_files:
            download_file(raw_dir, f)

        ghost_files = []
        try:
            ghost_files.append(find_file(my_analysis, reconstruction='axi', run=1, suffix='T2w', desc='segphantomMask', extension='.nii.gz')[0])
        except IndexError:
            gb._logprint(f'Could not find phantom mask')
        
        try:
            ghost_files.append(find_file(my_analysis, reconstruction='axi', run=1, suffix='T2w', desc='segFidLabelsUNetAxis', extension='.nii.gz')[0])
        except IndexError:
            gb._logprint(f'Could not find Fid label mask run 1')

        try:
            ghost_files.append(find_file(my_analysis, reconstruction='axi', run=2, suffix='T2w', desc='segFidLabelsUNetAxis', extension='.nii.gz')[0])
        except IndexError:
            gb._logprint(f'Could not find Fid label mask run 1')

        for f in ghost_files:
            download_file(os.path.join(deriv_dir, 'ghost'), f)

        csv_files = find_file(my_analysis, extension='.csv')

        for f in csv_files:
            download_file(os.path.join(deriv_dir, 'stats'), f)


def find_latest_analysis(analysis_list):
    correct_version = []
    
    for analysis in analysis_list:
        try:
            if analysis.gear_info.name == 'ghost':
                print("Found ghost analysis")
                correct_version.append(analysis)
        except:
            continue

    # Find latest
    time_stamps = [x.created for x in correct_version]
    latest_analysis = None
    try:
        latest_analysis = correct_version[ time_stamps.index(max(time_stamps)) ]    
        print(f"Using analysis {latest_analysis.label}")
    except:
        print(f"Found no ghost analysis")

    return latest_analysis


def find_file(analysis, reconstruction=None, run=None, desc=None, suffix=None, extension=None):
    
    match_str = "*"
    
    if reconstruction:
        match_str += f"_rec-{reconstruction}"

    if run:
        match_str += f'_run-{run:02d}'
    
    if desc:
        match_str += f'_desc-{desc}'

    if suffix:
        match_str += f'_{suffix}'

    if extension:
        match_str += extension
    
    all_files = [f.name for f in analysis.files]
    filenames = fnmatch.filter(all_files, match_str)

    out_files = []
    for file in analysis.files:
        if file.name in filenames:
            out_files.append(file)

    return out_files


def download_file(workdir, file, over_write=False):
    out_path = os.path.join(workdir, file.name)
    
    if not os.path.exists(out_path) and not over_write:
        file.download(out_path)

    return out_path


def get_dataset_ids(gear_context: GearToolkitContext, container):
    
    if container.container_type == 'project':
        output = {}

        for sub in container.subjects():
            sub_label = make_subject_label(sub)
            output[sub_label] = {'id':sub.id, 
                           'sessions':{}}

            for ses in sub.sessions():
                ses_label = make_session_label(ses)
                output[sub_label]['sessions'][ses_label] = ses.id

        return output

    elif container.container_type == 'subject':
        
        sub_label = make_subject_label(container)
        output = {sub_label:{'sessions':{}, 'id':container.id}}

        for ses in container.sessions():
            ses_label = make_session_label(ses)
            output[sub_label]['sessions'][ses_label] = ses.id
        
        return output

    elif container.container_type == 'session':

        sub = gear_context.client.get(container.parents.subject)
        sub_label = make_subject_label(sub)
        ses_label = make_session_label(container)

        output = {sub_label:{'id':sub.id, 'sessions':{ses_label:container.id}}}

        return output


def make_session_label(ses) -> str:
    return ses.label.split()[0].replace("-",'')


def make_subject_label(sub) -> str:
    return 'P'+sub.label.split('-')[1]


if __name__ == "__main__":  # pragma: no cover
    
    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        main(gear_context)