# GHOSTstats - FlyWheel gear ⚙️ 📈

![logo](doc/_static/ghost_logo.png)

## Running on FlyWheel

First run the [`fw-ghost`](https://github.com/UNITY-Physics/fw-GHOST) gear to produce the segmentation labels. This gear will then process the segmentations to get the stats as `.csv` files.

Similar to the [`fw-ghost`](https://github.com/UNITY-Physics/fw-GHOST) gear, this gear can run on Project, Subject or session level. It will calculate the stats labels and then produce a new analysis container where the results will be uploaded in BIDS naming format

## Build the gear

Check the `build_docker.sh` file which outlines how to build the gear, including downloading the necessary auxiliary files.

## Debug

This gear use the same debug method as in [`fw-ghost`](https://github.com/UNITY-Physics/fw-GHOST) with container IDs.
