#!/usr/bin/env python
"""The run script."""
import logging
import sys
import bids

# import flywheel functions
from flywheel_gear_toolkit import GearToolkitContext
from utils.parser import parse_config
from utils.command_line import exec_command
from ghost.bids import unity_qa_process_subject


# The gear is split up into 2 main components. The run.py file which is executed
# when the container runs. The run.py file then imports the rest of the gear as a
# module.

log = logging.getLogger(__name__)

def main(context: GearToolkitContext) -> None:
    """Parses config and runs."""
    input_dir, ouput_dir, sub, ses = parse_config(context)

    print("Indexing folder structure")
    print(index)

    if index:
        layout = bids.BIDSLayout(database_path=input_dir)
    else:
        layout = bids.BIDSLayout(root=input_dir, derivatives=ouput_dir)


    print("running main script...")

    # Set the command to be executed
    command =  unity_qa_process_subject(layout, sub, ses)

    # Execute the command
    exec_command(command)


# Only execute if file is run as main, not when imported by another module
if __name__ == "__main__":  # pragma: no cover
    # Get access to gear config, inputs, and sdk client if enabled.
    with GearToolkitContext() as gear_context:

        # Initialize logging, set logging level based on `debug` configuration
        # key in gear config.
        gear_context.init_logging()

        # Pass the gear context into main function defined above.
        main(gear_context)