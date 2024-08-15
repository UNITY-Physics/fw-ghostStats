# GHOST - Global Harmonisation Of Scanner performance Testing

Tools to process data acquired with the UNITY Phantom (Caliber Mini Hybrid Phantom).

## Install

Requires python3 but otherwise no special packages. Easiest way to install is to use `pip` inside a virtual/conda environment.

```sh
python -m pip install -e .
```

If you want to set up a new python environment with conda or pip you can use

```sh
conda create -n ghost python=3.9

```

There are some additional data files that are required such as phantom template data and deep learning models. All of this is easily downloaded by running the following command after installation

```sh
ghost setup
```

This will create a folder in your home directory called `ghost_data` which contains all the various phantom models.

### nnUNet for fiducial segmentation (Optional)

Segmentation of the geometric distortion fiducials is done using [nnUNet](https://github.com/MIC-DKFZ/nnUNet) which needs to be installed for this to be used. This is built into the container solutions (see below) but if you want to run this without a container you need to install nnUNet into your `ghost` python environment. I highly recommend going through the installation steps on the nnUNet website. But in brief it can be summarized as:

1. Install [pytorch](https://pytorch.org/get-started/locally/) using the instructions on the pytorch website.
2. Install nnUNet using `pip install nnunetv2`
3. Set up [environment variables for nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md). The following needs to be set

```sh
export nnUNet_raw="<my_path>/nnUNet_raw"
export nnUNet_preprocessed="<my_path>/nnUNet_preprocessed"
export nnUNet_results="<my_path>/nnUNet_results"
```

Then you need to install the nnUNet models. They are downloaded into the `nnUnet_models` folder in this repository. Once you have everything installed and the paths set up you navigate to the folder and run

```sh
nnUNetv2_install_pretrained_model_from_zip export237.zip
nnUNetv2_install_pretrained_model_from_zip export337.zip
nnUNetv2_install_pretrained_model_from_zip export437.zip
```

## Command line interface (CLI) usage

The GHOST package has a single binary which executes with the `ghost` command

```sh
usage: ghost <command> [<args>]

    Available commands are
    setup                    Download data
    warp_rois                Warp ROIS

```

- `setup`: First command to run which downloads the reference data used for segmentation
- `warp_rois`: Warp phantom segmentations/labels to your input data.

In general, the `ghost` command operates on file names on the command line. You can also call the underlying python functions which are found in `ghost.cmd`.

## UNITY QA BIDS analysis

For larger datasets it is convenint to have data organised in a BIDS structure. For this purpose there is a `ghost_unity_bids` tool which operates on a bids data structure and takes subject and session names as input. This is a work in progress.

See [UNITY QA description](examples/unity_QA/)

## Container

Recipes for building Docker and singularity/apptainer are provided here (`Dockerfile` and `apptainer.def`). After building this you will have the full 