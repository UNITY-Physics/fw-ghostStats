# GHOST - Global Harmonisation Of Scanner performance Testing

Tools to process data acquired with the UNITY Phantom (Caliber Mini Hybrid Phantom).

## Install
This is a simple pip package which you can install with
```
python -m pip install -e .
```

## Command line interface (CLI) usage
The GHOST package has a single binary which executes with the `ghost` command
```
usage: ghost <command> [<args>]

    Available commands are
	setup                    Download data
	update_sidecar           Update json sidecar info from dicom tags
	warp_rois                Warp ROIS

```

- `setup`: First command to run which downloads the reference data used for segmentation
- `update_sidecar`: Used to update sidecare info from dicom tags
- `warp_rois`: The main command used for segmenting phantom data.

## Conventions
- Analysis is performed in Python (version 3)
- Images are treated as `antspyx` image objects. This simplifies working with transformations, image arrays while maintaining information about geometry etc.
- To query BIDS datasets, we use [pybids](https://bids-standard.github.io/pybids/api.html#bids-layout-querying-bids-datasets). See 'GHOST_demo.ipynb' for short demonstration. Functions in 'bids.py' can be integrated in more thorough analysis.

## Converting DICOM library to BIDS-compatible NIFTI tree
dcm2bids converts DICOM files into a structured, standardized format that includes detailed metadata. See more about [BIDS](https://bids-specification.readthedocs.io/en/stable/02-common-principles.html).

### BIDS entities
- subject: site or phantom ID
- session: date of scan
- modality/suffix: T1w, T2w, dwi, PDw
- acquisition: std (standard), fast (T2w), gw (gray/white contrast, T1w), fisp (PDw)
- reconstruction: axi (axial), cor (coronal), sag (sagittal)
- datatype: anat, dwi
- run: ...

The data will then be saved in ```project/rawdata``` using this convention
```
sub-{subject}/
└─ses-{session}/
 └─{datatype<anat>|anat}/
        │ sub-{subject}_ses-{session}_acq-{acquisition}_rec-{reconstruction}_run-{run}_{suffix<T1w|T2w|FLAIR|dwi|PDw>}
          {extension<.nii|.nii.gz|.json>|.nii.gz}
```

### Example project setup guide:
Define a project directory with the following structure
```
project
│ README.md
│
└───code
│
└───derivatives
│  │ dataset_description.json
│
└───rawdata
│  │ dataset_description.json
│
└───sourcedata

```
1. Add DICOM data to `path/to/project/sourcedata/`
2. Copy the 'dcm2bids' folder in the GHOST repository to `path/to/project/code/`
3. Open run_dcm2bids.sh 
    - Read notes :)
    - Edit `PROJECT_DIR` and `PARTICIPANT_ID`
4. In the terminal, execute
    ```
    bash path/to/run_dcm2bids.sh
    ```

## Useful features
- Parse and plot information in the reference measurement sheet for the phantom ([Notebook example](phantom_characteristics.ipynb))

## Features to be implemented
- [ ] QC tool where the transformed segmentations are overlayed on the hyperfine image with text labels
- [ ] Source list of concentration in each vial. Check if these match between phantoms. Otherwise, input excell file and parse.
