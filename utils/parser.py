"""Parser module to parse gear config.json."""

from typing import Tuple
import flywheel
from flywheel_gear_toolkit import GearToolkitContext
import json
import os

from ghost.bids import setup_bids_directories, import_dicom_folder
from ghost.misc import ghost_path
from ghost.phantom import Caliber137



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
    gear_context: GearToolkitContext,
     
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
    input_id = (config['inputs']['input']['hierarchy']['id'])
    input_container = gear_context.client.get(input_id)

    print("API key is : ", api_key)
    # print("input_container: ", input_container)


    # Get the subject id from the session id
    # & extract the subject container
    subject_id = input_container.parents['subject']
    subject_container = gear_context.client.get(subject_id)
    subject = subject_container.reload()
    print("subject label: ", subject.label)
    subject_label = subject.label
    subject_label = subject_label.replace("-", "")


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
    # Import DICOMs. Use the example dataset
    config = '/flywheel/v0/examples/unity_QA/bids/dcm2bids_config.json'
    dicom_dir = input_dir #os.path.join(ghost_path(), "example_data/UNITY_QA/DICOM")
    sub_name = subject_label # For the phantom serial number in this case

    # Loop over the sessions
    for f in os.listdir(dicom_dir):
        p = os.path.join(dicom_dir, f)
        if os.path.isdir(p):
            import_dicom_folder(dicom_dir=p, sub_name=sub_name, ses_name=f, config=config, projdir=work_dir)


    return input_dir, ouput_dir, subject_label, session_label