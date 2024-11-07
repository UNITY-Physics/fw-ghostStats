"""Parser module to parse gear config.json."""

from typing import Tuple
import flywheel
from flywheel_gear_toolkit import GearToolkitContext
import json
import os
import zipfile
import shutil

from ghost.bids import setup_bids_directories, import_dicom_folder


def get_acq(session_container, session_label):
    download_dir = '/flywheel/v0/input/'
    try:
        for acq in session_container.acquisitions.iter():
            for file in acq.files:
                if file['type'] == 'dicom':
                    download_dir = ('/flywheel/v0/input/' + session_label)
                    if not os.path.exists(download_dir):
                        os.mkdir(download_dir)

                    download_path = download_dir + '/' + file.name
                    file.download(download_path)
                    print(f"Downloaded file: {file.name}")
        return True  # Completed successfully
    
    except Exception as e:
        print(f"Error downloading files: {e}")
        return False  # Completed unsuccessfully

def parse_config(
    gear_context: GearToolkitContext, input_id: str
     
) -> Tuple[str, str]: # Add dict for each set of outputs
    """Parse the config and other options from the context, both gear and app options.

    Returns:
        gear_inputs
        gear_options: options for the gear
        app_options: options to pass to the app
    """

    print("Running parse_config...")
    input_dir = '/flywheel/v0/input/'
    work_dir = '/flywheel/v0/work/'
    ouput_dir = '/flywheel/v0/output/'


    # Read config.json file
    p = open('/flywheel/v0/config.json')
    config = json.loads(p.read())

    # Read API key in config file
    api_key = (config['inputs']['api-key']['key'])

    if not input_id:

        try:
            input_id = (config['inputs']['input']['hierarchy']['id'])
        except KeyError:
            print("Can not find the hirerachy ID in the config. If you are running locally with a custom file input try to execute the run script as\n   python3 run.py CUSTOM_ID")
            

    input_container = gear_context.client.get(input_id)
    print(f"Input ID is: {input_id}", flush=True)


    # Get the subject id from the session id
    # & extract the subject container
    subject_id = input_container.parents['subject']
    subject_container = gear_context.client.get(subject_id)
    subject = subject_container.reload()
    print("subject label: ", subject.label)
    subject_label = subject.label
    subject_label = subject_label.replace("-", "")
    sub_name = subject_label # For the phantom serial number in this case

    # Get the session id from the input file id
    # & extract the session container
    session_id = input_container.parents['session']
    session_container = gear_context.client.get(session_id)
    session = session_container.reload()
    session_label = session.label
    date = session_label.split()[0] 
    ses_label = date.replace("-", "")

    print("session label: ", ses_label)
    
    # Download the dicom files from Flywheel
    get_acq(session_container, ses_label)

    # Create the BIDS directory structure
    setup_bids_directories(work_dir)

    bids_config = '/flywheel/v0/examples/unity_QA/bids/dcm2bids_config.json'
    dicom_dir = os.path.join(input_dir, ses_label)
    extracted_dicom_dir = os.path.join(dicom_dir, 'extracted')
    os.mkdir(extracted_dicom_dir)

    for f in os.listdir(dicom_dir):
        p = os.path.join(dicom_dir, f)
        if os.path.splitext(f)[1] == '.zip':
            zp = zipfile.ZipFile(p)
            members = zp.namelist()
            zp_path = zp.extract(members[0], path=dicom_dir)
            shutil.move(zp_path, extracted_dicom_dir)
    
    import_dicom_folder(dicom_dir=extracted_dicom_dir, sub_name=sub_name, ses_name=ses_label, config=bids_config, projdir=work_dir)

    return work_dir, ouput_dir, sub_name, ses_label, config