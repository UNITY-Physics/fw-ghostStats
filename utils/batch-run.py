
import os
from datetime import datetime
import pytz
import re
import pandas as pd
import math
import argparse
import flywheel
import logging

log = logging.getLogger(__name__)

def is_ghost_stats_analysis(analysis):
        """
        Check if an analysis is a ghost analysis by checking gear name or analysis label.
        """
        # Check gear name - must be exactly 'ghoststats' gear
        if analysis.label:
            label = analysis.label.lower()
            if "ghost/0.0.5" in label:
                return True
            
        return False

def main (fw):   
    
    gear =  fw.lookup('gears/ghost')
    analysis_tag = 'ghost'
    # Initialize gear_job_list
    job_list = list()

    
    fw_project = fw.lookup("unity/UNITY-QA")
    for subject in fw_project.subjects():
        subject = subject.reload()
        for session in subject.sessions():
            session = session.reload()
            # Check if a ghost analysis already exists for this session
            ghost_analyses = [analysis for analysis in session.analyses if is_ghost_analysis(analysis)]
            if ghost_analyses:
                print(f"Skipping session {session.label} - ghost analysis already exists.")
                continue
            try:
                # The destination for this analysis will be on the session
                dest = session
                time_fmt = '%d-%m-%Y_%H-%M-%S'
                analysis_label = f'{analysis_tag}_{datetime.now().strftime(time_fmt)}'
                job_id = gear.run(
                    analysis_label=analysis_label,
                    
                    destination=dest,
                    tags=["analysis", "ghost","gpu"],
                    config={
                    
                        }
                )
                job_list.append(job_id)
                print("Submitting Job: Check Jobs Log", dest.label)
            except Exception as e:
                print(f"WARNING: Job cannot be sent for {dest.label}. Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Update the CSV file with the latest data from the Flywheel UNITY QA project.')
    parser.add_argument('--apikey','-apikey',type=str,nargs='?',help='FW CLI API key')

    args = parser.parse_args()
    api_key = args.apikey

    fw = flywheel.Client(api_key=api_key)
    print(f"User: {fw.get_current_user().firstname} {fw.get_current_user().lastname}")
    main(fw)
