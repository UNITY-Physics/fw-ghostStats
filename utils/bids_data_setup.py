import os
import json
from pathlib import Path

def setup_bids_directory(input_dir, output_dir=None, dataset_name="My BIDS Dataset"):
    """
    Set up a BIDS-compliant directory structure and create necessary metadata files.
    
    Parameters:
    -----------
    input_dir : str
        Path to the input directory where the BIDS dataset will be created
    output_dir : str, optional
        Path to the derivatives directory for processed data
    dataset_name : str, optional
        Name of the dataset to be used in dataset_description.json
    
    Returns:
    --------
    tuple
        (input_path, output_path) as Path objects
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else None
    
    # Create input directory if it doesn't exist
    input_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset_description.json
    description = {
        "Name": dataset_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": [""],
        "Acknowledgements": "",
        "HowToAcknowledge": "",
        "Funding": [""],
        "ReferencesAndLinks": [""],
        "DatasetDOI": ""
    }
    
    with open(input_path / "dataset_description.json", "w") as f:
        json.dump(description, f, indent=4)
    
    # Create basic BIDS folder structure
    folders = [
        "code",
        "stimuli",
        "sourcedata",
    ]
    
    for folder in folders:
        (input_path / folder).mkdir(exist_ok=True)
    
    # Create derivatives directory if specified
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset_description.json for derivatives
        derivatives_description = {
            "Name": f"{dataset_name} derivatives",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": "Pipeline Name",
                "Version": "1.0.0",
                "Description": "Data processing pipeline"
            }],
            "SourceDatasets": [{
                "DOI": "",
                "URL": "",
                "Version": ""
            }]
        }
        
        with open(output_path / "dataset_description.json", "w") as f:
            json.dump(derivatives_description, f, indent=4)
    
    # Create README and CHANGES files
    with open(input_path / "README", "w") as f:
        f.write(f"# {dataset_name}\n\nDescription of your dataset goes here.")
    
    with open(input_path / "CHANGES", "w") as f:
        f.write("1.0.0 YYYY-MM-DD\n - Initial release")
        
    return input_path, output_path

def add_subject(input_dir, subject_id, sessions=None, tasks=None):
    """
    Add a subject directory structure to the BIDS dataset.
    
    Parameters:
    -----------
    input_dir : str or Path
        Path to the BIDS root directory
    subject_id : str
        Subject identifier (without 'sub-' prefix)
    sessions : list, optional
        List of session identifiers
    tasks : list, optional
        List of task names
    """
    input_path = Path(input_dir)
    subject_dir = input_path / f"sub-{subject_id}"
    subject_dir.mkdir(exist_ok=True)
    
    # Define common datatypes
    datatypes = ['anat', 'func', 'dwi', 'fmap']
    
    if sessions:
        # Create session-level directories
        for session in sessions:
            session_dir = subject_dir / f"ses-{session}"
            session_dir.mkdir(exist_ok=True)
            
            # Create datatype directories within session
            for datatype in datatypes:
                (session_dir / datatype).mkdir(exist_ok=True)
    else:
        # Create subject-level datatype directories
        for datatype in datatypes:
            (subject_dir / datatype).mkdir(exist_ok=True)