import json
import logging
import os
from string import ascii_lowercase as alc
import sys
from typing import Tuple

from flywheel_gear_toolkit import GearToolkitContext

from ghost.bids import import_dicom_folder, setup_bids_directories

log = logging.getLogger(__name__)


def parse_config(gear_context: GearToolkitContext, input_id: str) -> Tuple[str, str]:
    
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
    config['bids_config_file'] = base_dir + '/examples/unity_QA/bids/dcm2bids_config.json'

    return container, config, manifest, inputs


def download_dataset(gear_context: GearToolkitContext, container, config):
    
    work_dir = config['work_dir']
    source_data_dir = os.path.join(work_dir, 'sourcedata')
    os.makedirs(source_data_dir, exist_ok=True)
    
    setup_bids_directories(work_dir)
    import_options = {'config': config['bids_config_file'], 'projdir': work_dir, 'skip_dcm2niix': True}

    if container.container_type == 'project':
        proj_label, subjects = download_project(container, source_data_dir, dry_run=False)

        output = {}
        for sub in subjects.keys():
            output[sub] = {}
            sessions = subjects[sub]

            for ses in sessions.keys():
                import_dicom_folder(dicom_dir=subjects[sub][ses]['folder'], sub_name=sub, ses_name=ses, **import_options)
                output[sub][ses] = subjects[sub][ses]['id']

        return output

    elif container.container_type == 'subject':
        proj_label = gear_context.client.get(container.parents.project).label
        source_data_dir = os.path.join(source_data_dir, proj_label)
        
        sub_label, sessions = download_subject(container, source_data_dir, dry_run=False)
        
        output = {sub_label:{}}

        for ses in sessions.keys():
            import_dicom_folder(dicom_dir=sessions[ses]['folder'], sub_name=sub_label, ses_name=ses, **import_options)
            output[sub_label][ses] = sessions[ses]['id']
        
        return output

    elif container.container_type == 'session':
        proj_label = gear_context.client.get(container.parents.project).label
        sub_label = make_subject_label(gear_context.client.get(container.parents.subject))
        source_data_dir = os.path.join(source_data_dir, proj_label, sub_label)
        
        ses_label, ses_dir, ses_id = download_session(container, source_data_dir, dry_run=False)

        import_dicom_folder(dicom_dir=ses_dir, sub_name=sub_label, ses_name=ses_label, **import_options)

        return {sub_label: {ses_label: ses_id}}


def make_session_label(ses) -> str:
    return ses.label.split()[0].replace("-",'')


def make_subject_label(sub) -> str:
    return 'P'+sub.label.split('-')[1]


def download_file(file, my_dir, dry_run=False) -> str:
    do_download = False

    if file['type'] in ['source code','nifti']:
        if ('T2' in file.name) and any(x in file.name.lower() for x in ['cor_fast', 'sag_fast', 'axi_fast']):
            do_download = True
    
    if do_download:
        download_dir = my_dir
        os.makedirs(download_dir, exist_ok=True)
        
        try:
            if dry_run:
                print(f"[DRY RUN] Would have downloaded: {file.name}")
            else:
                fpath = os.path.join(download_dir, file.name)
                if not os.path.exists(fpath):
                    file.download(fpath)
                else:
                    print("File already downloaded")

        except Exception as e:
            print(f"Error downloading {file.name}: {e}")

        print(f"Downloaded file: {file.name}")

    else:
        print(f"Skipping file: {file.name}")

    return file.name


def download_session(ses_container, sub_dir, dry_run=False) -> Tuple[str, str]:
    print("--- Downloading subject ---")
    print(f"Session label: {ses_container.label}")
    print(f"Sessions: {len(ses_container.acquisitions())}")

    ses_label = make_session_label(ses_container)
    ses_dir = os.path.join(sub_dir, ses_label)
    ses_id = ses_container.id
    print(f"Saving data into: {ses_dir}")

    for acq in ses_container.acquisitions.iter():
        for file in acq.files:
            download_file(file, ses_dir, dry_run=dry_run)

    return ses_label, ses_dir, ses_id


def download_subject(sub_container, proj_dir, dry_run=False):
    print("--- Downloading subject ---")
    print(f"Label: {sub_container.label}")
    print(f"Sessions: {len(sub_container.sessions())}")
    
    sub_label = make_subject_label(sub_container)
    sub_dir = os.path.join(proj_dir, sub_label)
    print(f"Saving data into: {sub_dir}")
    
    sessions_out = {}

    for ses in sub_container.sessions.iter():
        ses_label0, ses_dir, ses_id = download_session(ses, sub_dir, dry_run=dry_run)

        # Check for duplicate session labels
        ses_label = ses_label0; i = 0
        
        while ses_label in sessions_out:
            ses_label = ses_label0 + alc[i]
            i += 1
        

        sessions_out[ses_label] = {'folder':ses_dir, 'id':ses_id}

    return sub_label, sessions_out


def download_project(project, my_dir, dry_run=False):
    print("--- Downloading project ---")
    print(f"Label: {project.label}")
    print(f"Subjects: {project.stats.number_of.subjects}")
    print(f"Sessions: {project.stats.number_of.sessions}")
    print(f"Acquisitions: {project.stats.number_of.acquisitions}")
    
    proj_name = project.label
    my_dir = os.path.join(my_dir, proj_name)
    print(f"Saving data into: {my_dir}")
    
    subjects_out = {}
    for sub in project.subjects.iter():
        sub_lab, sessions_dict = download_subject(sub, my_dir, dry_run=dry_run)
        subjects_out[sub_lab] = sessions_dict

    return proj_name, subjects_out
