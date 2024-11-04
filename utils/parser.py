"""Parser module to parse gear config.json."""

from typing import Tuple
import flywheel
from flywheel_gear_toolkit import GearToolkitContext
import json


def get_acq(session_container):
    download_dir = '/flywheel/v0/input/'
    try:
        for acq in session_container.acquisitions.iter():
            for file in acq.files:
                if file['type'] == 'nifti':
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

    # Get the session id from the input file id
    # & extract the session container
    session_id = input_container.parents['session']
    session_container = gear_context.client.get(session_id)
    session = session_container.reload()
    session_label = session.label
    print("session label: ", session.label)
    
    get_acq(session_container)

    input_dir = '/flywheel/v0/input/'
    ouput_dir = '/flywheel/v0/output/'

    return input_dir, ouput_dir, subject_label, session_label