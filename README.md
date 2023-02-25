# GHOST - Global Harmonisation Of Scanner performance Testing

Tools to process data acquired with the UNITY Phantom (Caliber Mini Hybrid Phantom).

## Install
This is a simple pip package which you can install with
```
python -m pip install -e .
```

To download phantom reference data run
```
ghost_download_phantom
```
or in a python environment
```
from ghost.download_data import download_phantom

download_phantom()
```

## Conventions
- Analysis is performed in Python (version 3)
- Images are treated as `antspyx` image objects. This simplifies working with transformations, image arrays while maintaining information about geometry etc.


## Useful features
- Parse and plot information in the reference measurement sheet for the phantom ([Notebook example](phantom_characteristics.ipynb))

## Features to be implemented
- [ ] A standard geometrically accurate template with everything segmented
- [ ] Automatically resample the phantom to the acquired resolution of the hyperfine scan
- [ ] Automatic registration and transformation of labels to hyperfine image
- [ ] Assessment of linear and non-linear warp field to template image
- [ ] QC tool where the transformed segmentations are overlayed on the hyperfine image with text labels
- [ ] Source list of concentration in each vial. Check if these match between phantoms. Otherwise, input excell file and parse.
