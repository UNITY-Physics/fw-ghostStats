# GHOST - Global Harmonisation Of Scanner performance Testing

Tools to process data acquired with the UNITY Phantom (Caliber Mini Hybrid Phantom).

## Install

Install the package and download the dependencies

```sh
python -m pip install -e .
ghost setup
```

## Command line interface (CLI) usage

The GHOST package has a single binary which executes with the `ghost` command

```sh
usage: ghost <command> [<args>]

    Available commands are
    setup                    Download data
    update_sidecar           Update json sidecar info from dicom tags
    warp_rois                Warp ROIS

```

- `setup`: First command to run which downloads the reference data used for segmentation
- `update_sidecar`: Used to update sidecare info from dicom tags
- `warp_rois`: The main command used for segmenting phantom data.

## UNITY QA BIDS analysis

See [examples/unity_QA/](UNITY QA description)

## Conventions

- Analysis is performed in Python (version 3)
- Images are treated as `antspyx` image objects. This simplifies working with transformations, image arrays while maintaining information about geometry etc.
- To query BIDS datasets, we use [pybids](https://bids-standard.github.io/pybids/api.html#bids-layout-querying-bids-datasets). See 'GHOST_demo.ipynb' for short demonstration. Functions in 'bids.py' can be integrated in more thorough analysis.

## Containerized

In progress. Watch this space...