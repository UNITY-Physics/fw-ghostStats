"""Parser module to parse gear config.json."""

from typing import Tuple
import flywheel
from flywheel_gear_toolkit import GearToolkitContext
import pandas as pd
import os
import json
import re
from datetime import datetime

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
    input = gear_context.get_input_path("input")
    input_dir = '/flywheel/v0/input/'
    ouput_dir = '/flywheel/v0/output/'

    # Read config.json file
    p = open('/flywheel/v0/config.json')
    config = json.loads(p.read())

    # Read API key in config file
    api_key = (config['inputs']['api-key']['key'])
    fw = flywheel.Client(api_key=api_key)
    
    # Get the input file id
    input_container = gear_context.client.get_analysis(gear_context.destination["id"])

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
    
    return input_dir, ouput_dir, subject_label, session_label