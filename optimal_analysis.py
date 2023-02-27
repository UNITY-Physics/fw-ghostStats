# Pseudo code for how to run our analysis in python

import ghost

# Read in our images as ants objects
my_swoop_img = ants.read_imag('<fname>')
my_ref_img = ants.read_img(get_phantom_nii('T1'))

# Run registration and get transform object
xfm = reg_to_phantom(my_swoop_img, my_ref_img, xfm_type='Affine')

# Can be nice to have this saved
save_xfm(xfm, fname)

# Apply transformation to given segmentation
T1_seg = warp_seg(ref=my_swoop_img, xfm=xfm, which='T1')

# Will be a useful function
save_seg(T1_seg, fname)

# Calculate slice thickness
# This script does all the warping of wedge segmentations in it already
calc_slicethickness(my_swoop_img)

# Parse the ROI values, get mean and sd. Labeled array with ID, mean, SD
T1_rois_vals = parse_rois(my_swoop_img, T1_seg)

# Parse calibration values. Make this into an object which is initiated with this function.
# Then we can call functions on this object to get specific data values
calib = read_calibration_data(fname='<calib_file.xls>')

# Concentration is the same at each temperature. T1 is the NiCl
T1_conc = calib.get_T1_conc()
T2_conc = calib.get_T2_conc()

# Get T1 or T2 values at a specific temperature or field strength. 
# Throw warning if trying to get B0 or temp values that are out of range
qT1_vals = calib.get_T1_vals(B0=3, temp=20)

#################################################################
# We can also envision this done with one function

my_swoop_fname = '<fname>'
contrast = 'T1'
output_dir = '<my_dir>'
output_basename = 'swoop_T1'

process_all(my_swoop_fname, contrast, output_basename, output_dir)
# This will then create a new folder (if needed) and save segmentations and xfm as
# output_dir/output_basename_T1 / T2 / ADC etc.
# This script could then easily be wrapped into a bash script