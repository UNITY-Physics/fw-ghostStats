# UNITY QA Analysis (BIDS style)

## 1. DICOM Import

Assuming that you have your dicoms in a nice folder stucture with some naming convention for session and subject you can import your DICOMs to BIDS format like this:

```python
from ghost.bids import import_dicom_folder, setup_bids_directories

projdir = '/myproject
dcm2bids_config = 'dcm2bids_config_87.json'

setup_bids_directories(projdir)
import_dicom_folder(dicom_dir, sub_name, ses_name, dcm2bids_config, projdir)
```

This will give you a project directory that follows a typical BIDS structure and is BIDS compliant.

## 2. Run the pipeline

The main analysis tools are in `ghost.phantom` and `ghost.bids`. The idea is that functions that operate on a file level basis are in `ghost.phantom` while `ghost.bids` operate on a session level (this isn't really that organized yet though). Have a look in `ghost.bids.unity_qa_process_subject` for the steps that we run. To execute this for a single subject:

```python
import bids
from ghost.bids import unity_qa_process_subject

ses = session
sub = 'subject
projdir = '/myproject'
layout = bids.BIDSLayout(root=projdir+'/rawdata', derivatives=projdir+'/derivatives')

unity_qa_process_subject(layout, sub, ses)
```

The `layout` object is really handy to work with since it will give us keyword access to filenames in our project and throw errors if they don't exist.

## 3. Review the results

Have a look at the `plotting.ipynb` for some examples of visualizing the data. The `stats` folder does contain numerical results as well that can be parsed. Baring in mind that there might be some bugs in there still.

## Other

The notebook `describe_fid_seg.ipynb` goes through the steps of the fiducial segmentation.