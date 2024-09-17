# GHOST - Global Harmonisation Of Scanner performance Testing

![logo](doc/_static/ghost_logo.png)

Tools to process data acquired with the UNITY Phantom (Caliber Mini Hybrid Phantom).

## Install

Requires python3 but otherwise no special packages. Easiest way to install is to use `pip` inside a virtual/conda environment.

```sh
python -m pip install -e .
```

If you want to set up a new python environment (recommended) with conda or pip you can use

```sh
conda create -n ghost python=3.9
```

There are some additional data files that are required such as phantom template data and deep learning models. All of this is easily downloaded by running the following command after installation (remember to activate your environment!). 

```sh
ghost setup --phantoms
```

This will create a folder in your home directory called `ghost_data` which contains all the various phantom models.

To build the documentation, run the following:

```sh
python3 make_doc.py
```

You find the documentation by opening the [`index.html`](doc/_build/index.html) in your web browser.

### nnUNet for fiducial segmentation (Optional)

Segmentation of the geometric distortion fiducials is done using [nnUNet](https://github.com/MIC-DKFZ/nnUNet) which needs to be installed for this to be used. This is built into the container solutions (see below) but if you want to run this without a container you need to install nnUNet into your `ghost` python environment. Execute the following steps:

1. Install [pytorch](https://pytorch.org/get-started/locally/) using the instructions on the pytorch website.
2. Install nnUNet using `pip install nnunetv2`

Once you have this set up, you can download and import the pre-trained nnUNet models

```sh
ghost setup --nnUNet
```

This command will also import the models into the `ghost_data` directory. The `nnUNet` library typically requires you to set dedicated system paths for where to find the pre-trained models. In `ghost` we set these at run time to be the `ghost_data/nnUNet` directory to avoid clashes with your local setup.

## Command line interface (CLI) usage

The GHOST package has a single binary which executes with the `ghost` command

```sh
usage: ghost <command> [<args>]

    Available commands are
    setup                    Download data
    warp_rois                Warp ROIS

```

- `setup`: First command to run which downloads the reference data used for segmentation (options `--phantoms`, `--examples`, `--nnUnet`)
- `warp_rois`: Warp phantom segmentations/labels to your input data.

In general, the `ghost` command operates on file names on the command line. You can also call the underlying python functions which are found in `ghost.cmd`.

## Python api

In addition to the command line interface it is very easy to interact with `ghost` directly in python. Have a look at the example notebooks or the documentation for more information.

## Examples

A [sample dataset](https://figshare.com/articles/dataset/UNITY_Phantom_QA_example_data/26954056) is provided for local testing. You can download this to your `ghost_data` directory using

```sh
ghost setup --examples
```

The two examples below uses this example dataset.

### UNITY QA BIDS analysis

For larger datasets it is convenint to have data organised in a BIDS structure. For this purpose there is a `ghost_unity_bids` tool which operates on a bids data structure and takes subject and session names as input. This is a work in progress.

See [UNITY QA description](examples/unity_QA/)

## Container

Recipes for building a Docker container is provided in the Dockerfile. Use the `build_docker.sh` to build the docker container, the installation of `torch` is different depending on your platform.
