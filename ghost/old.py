def calc_atropos_fiducials(img_in, fiducials, fiducial_mask, BG):
    swoopT2w = ants.resample_image(img_in, resample_params=[1, 1, 1], use_voxels=False, interp_type=4)
    
    print('Co-registering to high res template')
    xfm = reg_to_phantom(swoopT2w, phantom_weighting='T2')

    print("Warping masks to interpolated swoop space")
    fid_warp = ants.apply_transforms(fixed=swoopT2w, moving=fiducials, transformlist=xfm, whichtoinvert=[True, False], interpolator='genericLabel')
    fid_mask_warp = ants.apply_transforms(fixed=swoopT2w, moving=fiducial_mask, transformlist=xfm, whichtoinvert=[True, False], interpolator='genericLabel')
    bg_warp = ants.apply_transforms(fixed=swoopT2w, moving=BG, transformlist=xfm, whichtoinvert=[True, False], interpolator='genericLabel')

    images = []
    masks = []
    mask_data = fid_warp.numpy()

    print("Pre-processing masks")
    for i in range(1,16):
        new_data = np.zeros_like(mask_data)
        new_data[mask_data==i] = 1
        new_img = ants.from_numpy(new_data, origin=fid_warp.origin, spacing=fid_warp.spacing, direction=fid_warp.direction)
        new_img = ants.smooth_image(new_img, sigma=2)
        images.append(new_img)

        masks.append(ants.threshold_image(fid_mask_warp, i-0.1, i+0.1))

    bg_mask = ants.smooth_image(bg_warp, sigma=2)

    print('Calculating atropos segmentation')
    segmentations = []
    for i in range(15):
        seg = ants.prior_based_segmentation(swoopT2w, [bg_mask, images[i]], masks[i], iterations=2, priorweight=0.5, mrf=0.2)
        segmentations.append(seg)
    
    print("Merging segmentations")
    seg_out = np.zeros_like(fid_warp.numpy())
    for i in range(15):
        my_seg = segmentations[i]['segmentation'].numpy()
        seg_out[my_seg==2] = i+1

    seg_out_img = ants.from_numpy(seg_out, origin=fid_warp.origin, spacing=fid_warp.spacing, direction=fid_warp.direction)

    return swoopT2w, seg_out_img, xfm

def calc_temperature(img, lc_mask, T1_mask, T2_mask, bg_mask, fname_png=None):
    
    img = ants.denoise_image(img)
    img_np = img.numpy()
    lc_np = lc_mask.numpy()
    bg_np = bg_mask.numpy()
    T1_np = T1_mask.numpy()
    T2_np = T2_mask.numpy()

    LC_data = [np.mean(img_np[lc_np==i]) for i in range(1,11)]
    T1_data = [np.mean(img_np[T1_np==i]) for i in range(1,15)]
    T2_data = [np.mean(img_np[T2_np==i]) for i in range(1,15)]
    
    BG = np.mean(img_np[bg_np==1])
    # LC_data = np.array([BG]*5 + LC_data + [(T1_data[0]+T2_data[0])/2])
    LC_data = np.array([BG]*5 + LC_data)
    LC_data /= np.max(LC_data)
    x = np.arange(-5,10)

    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return (y)

    popt, pcov = curve_fit(sigmoid, x, LC_data, [1,5,1,min(LC_data)], method='dogbox')
    estimated_temperature = np.floor(15 + popt[1])
    
    if fname_png:
        plt.style.use('dark_background')
        cog = center_of_mass(lc_np)
        fig, axes = plt.subplots(1,3,figsize=(8,4))
        axes[0].imshow(np.fliplr(np.rot90(img_np[:,:,int(cog[2])])), cmap='gray'); axes[0].axis('off')

        lc_np[lc_np==0] = np.nan
        axes[1].imshow(np.fliplr(np.rot90(img_np[:,:,int(cog[2])])), cmap='gray'); axes[1].axis('off')
        axes[1].imshow(np.fliplr(np.rot90(lc_np[:,:,int(cog[2])])), cmap='jet')

        axes[2].plot(x,LC_data, 'o')
        axes[2].plot(x,sigmoid(x, popt[0], popt[1], popt[2], popt[3]), '-')
        
        plt.suptitle(f'Estimated temperature {estimated_temperature}C')
        plt.savefig(fname_png)
        plt.close()

    return estimated_temperature

def calculate_slice_thickness_from_wedges(img_data, seg_data, sigma=3, wedge_angle=10, resolution=None, return_plot_data=False):
    """Calculate the slice thickness of a 3D image. The slice thickness is calculated by fitting a Gaussian to the intensity gradient along the wedges of the image and calculating the slice thickness from the standard deviation of the Gaussian.
    
    Parameters
    ----------
    nifti_image : nifti image or numpy array
        The 3D image to calculate the slice thickness of.

    nifti_segmentation : nifti image or numpy array
        The segmentation of the 3D image indicating the location of the wedges. The segmentation of the wedges should be labelled 1 and 2.

    sigma : int, optional
        The standard deviation of the Gaussian window used to smooth the data. The default is 3.

    wedge_angle : int, optional
        The angle of the wedge in degrees. The default is 10.

    resolution : int, optional
        The resolution of the image in mm. Neccessary if image data is numpy.array The default is None.

    return_plot_data : bool, optional
        Whether to return the data needed to plot the slice thickness distributions. The default is False.
    
    Returns
    -------
    slice_thickness : int
        The slice thickness of the 3D image.

    w : int
        The projected wedge thicknesses.

    theta : int
        The angle of the image plane with respect to the base of the wedges.

    plot_data : dict
        The data needed to plot the slice thickness distributions. Only returned if return_plot_data is True.
    """
    
    if isinstance(img_data, (nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image)):
        # Get the image data and resolution
        image = img_data.get_fdata()
        head = img_data.header
        resolution = head.get('pixdim')[1:4]
    
    elif isinstance(img_data, np.ndarray):
        # Get the image data and resolution
        image = img_data
        if resolution is None:
            raise TypeError("Resolution is not defined. Please define the resolution of the image in mm.")
    
    if isinstance(seg_data, (nib.nifti1.Nifti1Image, nib.nifti2.Nifti2Image)):
        # Get the wedge segmentations
        seg1 = seg_data.get_fdata() == 1
        seg2 = seg_data.get_fdata() == 2
    
    elif isinstance(seg_data, np.ndarray):
        seg1 = seg_data == 1
        seg2 = seg_data == 2

    intensities = []
    wi = [] #Wedge index
    for seg in [seg1, seg2]:
        id = np.where(seg)
        min_i, max_i = np.min(id[0]), np.max(id[0])
        min_j, max_j = np.min(id[1]), np.max(id[1])
        min_k, max_k = np.min(id[2]), np.max(id[2])
        wi.append([[min_i, max_i], [min_j, max_j], [min_k, max_k]])

    y0 = max(wi[0][1][0], wi[1][1][0])
    y1 = min(wi[0][1][1], wi[1][1][1])
    wi[0][1][0] = y0
    wi[1][1][0] = y0
    wi[0][1][1] = y1
    wi[1][1][1] = y1

    for i,seg in enumerate([seg1, seg2]):
        id = np.where(seg)
        yz_planes = image[wi[i][0][0]:wi[i][0][1]+1, wi[i][1][0]:wi[i][1][1]+1, wi[i][2][0]:wi[i][2][1]+1] # yz
        slice_id = np.arange(np.min(id[2]),np.max(id[2])+1)
        intensities.append(np.mean(yz_planes, axis=0))

    indices = []
    for intensity in intensities:
        mean_intensity_y = np.mean(intensity, axis=0)
        index_mean = np.argmin(np.abs(intensity - mean_intensity_y), axis=0)
        percentage_mean = (index_mean / len(intensity)) * 100
        indices.append(np.argmin(np.abs(50 - percentage_mean)))
    
    # Make sure the wedges are segmented correctly
    if indices[0] == indices[1]:
        index = indices[0]
    else:
        index = int(np.floor((indices[0]+indices[1])/2))
    # else:
        # raise TypeError("Index of wedges do not match. Check segmentation of the wedges.")
    
    slice = slice_id[index]

    lp1, lp2 = intensities
    lp = np.stack((lp1[:,index],lp2[:,index])) # line profile

    # Filter the data with a Gaussian window to make it differentiable
    lp_smooth = ndi.gaussian_filter1d(lp, sigma, 1)

    # Differentiate the data with respect to image position
    npad = 10
    lp_diff = np.zeros((2,lp_smooth.shape[1]+2*npad))
    for i in range(lp_smooth.shape[0]):
        lp_diff[i,:] = np.pad(abs(np.gradient(lp_smooth[i,:])), npad)

    x_axis = np.arange(lp_diff.shape[1])*resolution[1]

    # define the gaussian function to fit
    def gauss(x, a, b, c):
        return a * np.exp(-((x-b)**2/(2*c**2)))
    
    lp_fit = np.empty((lp_diff.shape[0], lp_diff.shape[1]))
    w = np.empty(2) # projected wedge thicknesses
    for i in range(lp_diff.shape[0]):
        # Perform the curve fit
        popt, pcov = curve_fit(gauss, x_axis, lp_diff[i], p0=[1,np.median(x_axis),1], bounds=([0,np.min(x_axis),0], [np.inf, np.max(x_axis), np.inf]))
        
        # Calculate the fitted curves
        lp_fit[i] = gauss(x_axis, *popt)

        # Compute the projected wedge thicknesses
        w[i] = popt[2]*2*np.sqrt(2*np.log(2))

    # Calculate theta (the angle of the image plane with respect to the base of the wedges)
    theta = 1 / 2 * np.arcsin((w[1] - w[0]) * np.sin(2 * np.radians(10)) / sum(w))

    # Get the slice thickness
    slice_thickness = w[0]  * np.tan(np.radians(wedge_angle) + theta)

    if return_plot_data:
        # Make dictionary of variables needed to plot the slice thickness distributions
        plot_data = {'lp': lp, 
                     'lp_smooth': lp_smooth, 
                     'lp_diff': lp_diff, 
                     'lp_fit': lp_fit, 
                     'x_diff': x_axis, 
                     'x_lp': np.arange(lp_smooth.shape[1])*resolution[1],
                     'seg1': seg1, 
                     'seg2': seg2, 
                     'slice': slice}
        return slice_thickness, w, theta, plot_data
    else:
        return slice_thickness, w, theta

def transform_reference_segmentation(ref_img_path, ref_seg_path, target_img_path):
    """Register the target image to the reference image and apply the transformation to the reference segmentation.
    
    Parameters
    ----------
    ref_img_path : str
        Path to the reference image.
    
    ref_seg_path : str
        Path to the reference segmentation.
        
    target_img_path : str
        Path to the target image.
        
    Returns
    -------
    img_data : numpy.ndarray
        The target image data.
    
    seg_data : numpy.ndarray
        The transformed reference segmentation."""

    ref = ants.image_read(ref_img_path)
    seg = ants.image_read(ref_seg_path)
    target = ants.image_read(target_img_path)

    reg = ants.registration(fixed=ref, moving=target, type_of_transform='Affine')
    seg_warp = ants.apply_transforms(fixed=target, moving=seg, whichtoinvert=[1], transformlist=reg['fwdtransforms'], interpolator='genericLabel')

    img_data = target.numpy()
    seg_data = seg_warp.numpy()

    return img_data, seg_data

def save_xfm(xfm, filename):
    """Save the transformation object to a file
    
    Parameters
    ----------
    xfm : ANTsTransform
        The transformation object.
    
    filename : str
        Filename of transform (file extension is ".mat" for affine transforms).
    """
    ants.write_transform(xfm, filename)

def generate_masks():
    parser = argparse.ArgumentParser(description='Generate T1, T2, and ADC masks from a given input file')
    parser.add_argument('input_file', help='input file path')
    parser.add_argument('--ref', default='T1' , help='reference image for registration (T1 or T2)', type=str)
    parser.add_argument('--output_prefix', help='prefix for output files', type=str)
    parser.add_argument('--seg', default='all', help='segmentation image for registration (T1 or T2 or ADC)', type=str)
    args = parser.parse_args()

    input_file = args.input_file
    ref = args.ref
    output_prefix = args.output_prefix if args.output_prefix else os.path.splitext(os.path.basename(input_file))[0]

    if args.seg == 'all':
        seg = ['T1', 'T2', 'ADC']
    elif args.seg == 'T':
        seg = ['T1', 'T2']
    else:
        seg = [args.seg]

    # use input_file and output_prefix in your function code
    target_img = ants.image_read(input_file)

    # Find out if target_ANTsImage is T1w or T2w by looking at the metadata
    ref_img = ants.image_read(get_phantom_nii(ref))
    reg = ants.registration(fixed=ref_img, moving=target_img, type_of_transform='Affine')
    xfm = reg['fwdtransforms']

    for s in seg:
        seg_bin = ants.image_read(get_phantom_nii(s))
        warped_seg = ants.apply_transforms(fixed=target_img, moving=seg_bin, transformlist=xfm)
        # save each warped seg image to a file as output_prefix_seg.nii.gz

        # print "Created output_prefix_T1mask.nii.gz"
        print("Created " + output_prefix + '_' + s + 'mask.nii.gz')