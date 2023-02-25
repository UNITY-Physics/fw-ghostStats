import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def gauss(x, a, b, c):
    """Gaussian fit function"""
    return a * np.exp(-((x-b)**2/(2*c**2)))

def get_line_profiles(image, seg1, seg2):
        """ For clearer acquiring of the line profile of each wedge"""

        def get_intensities(image, seg):
            """ Get the mean intensity along the x-axis for each yz-plane"""
            id = np.where(seg)
            min_i, max_i = np.min(id[0]), np.max(id[0])
            min_j, max_j = np.min(id[1]), np.max(id[1])
            min_k, max_k = np.min(id[2]), np.max(id[2])
            planes = image[min_i:max_i+1, min_j:max_j+1, min_k:max_k+1] # yz
            slice_id = np.arange(np.min(id[2]),np.max(id[2])+1)
            intensities = np.mean(planes, axis=0)
            
            return intensities, slice_id

        def find_index(intensity_y):
            mean_intensity_y = np.mean(intensity_y, axis=0)
            index_mean = np.argmin(np.abs(intensity_y - mean_intensity_y), axis=0)
            percentage_mean = (index_mean / len(intensity_y)) * 100
            index = np.argmin(np.abs(50 - percentage_mean))
            return index
        
        intensities_y1, slice_id = get_intensities(image, seg1)
        intensities_y2, _ = get_intensities(image, seg2)

        index1 = find_index(intensities_y1)
        index2 = find_index(intensities_y2)

        # Make sure the wedges are segmented correctly
        if index1 == index2:
            index = index1
        else:
            raise TypeError("Something appears to be wrong with the segmentation of the wedges.")
        slice = slice_id[index]

        lp = np.array([intensities_y1[:,index],intensities_y2[:,index]])    

        return lp, slice

def get_slice_thickness(nifti_image, nifti_segmentation, sigma=3, wedge_angle=10, plot=False):
    """Calculate the slice thickness of a 3D image. The slice thickness is calculated by fitting a Gaussian to the intensity gradient along the wedges of the image and calculating the slice thickness from the standard deviation of the Gaussian.
    
    Parameters
    ----------
    nifti_image : nifti image
        The 3D image to calculate the slice thickness of.

    nifti_segmentation : nifti image
        The segmentation of the 3D image indicating the location of the wedges. The segmentation of the wedges should be labelled 1 and 2.

    sigma : int, optional
        The standard deviation of the Gaussian window used to smooth the data. The default is 3.

    wedge_angle : int, optional
        The angle of the wedge in degrees. The default is 10.

    plot : bool, optional
        Whether to plot the slice thickness distribution. The default is False.
    
    Returns
    -------
    slice_thickness : int
        The slice thickness of the 3D image.

    w1 : intarray
        Projected thicknesses of wedge 1.

    w2 : intarray
        Projected thicknesses of wedge 2.

    theta : int
        The angle of the image plane with respect to the base of the wedges.
    """

    # Get the image data and resolution
    image = nifti_image.get_fdata()
    head = nifti_image.header
    resolution = head.get('pixdim')[1:4]

    # Get the wedge segmentations
    seg1 = nifti_segmentation.get_fdata() == 1
    seg2 = nifti_segmentation.get_fdata() == 2

    lp, slice = get_line_profiles(image, seg1, seg2)

    x_axis = np.arange(lp.shape[1])*resolution[1]

    # Filter the data with a Gaussian window to make it differentiable
    lp_gauss = ndi.gaussian_filter1d(lp, sigma, axis=0)

    # Differentiate the data with respect to image position
    d_lp = np.abs(np.gradient(lp_gauss, x_axis, axis=0))

    bounds = ([0,-np.inf,0], [np.inf, np.inf, np.inf])
    p0 = [1,0,1]
    coeff_lp1, var_matrix = curve_fit(gauss, x_axis, d_lp[0], p0=p0, bounds=bounds)
    coeff_lp2, var_matrix = curve_fit(gauss, x_axis, d_lp[1], p0=p0, bounds=bounds)
    d_lp_fit = np.array([gauss(x_axis, *coeff_lp1),gauss(x_axis, *coeff_lp2)])

    # Get the projected wedge thicknesses
    w1 = coeff_lp1[2]*2*np.sqrt(2*np.log(2))
    w2 = coeff_lp2[2]*2*np.sqrt(2*np.log(2))

    # Calculate theta (the angle of the image plane with respect to the base of the wedges)
    theta = 1 / 2 * np.arcsin((w2 - w1) * np.sin(2 * np.radians(10)) / (w2 + w1))

    # Get the slice thickness
    slice_thickness = w1  * np.tan(np.radians(wedge_angle) + theta)

    res = dict()
    res['slice_thickness'] = slice_thickness
    res['w1'] = w1
    res['w2'] = w2
    res['theta'] = theta
    res['lp'] = lp
    res['lp_gauss'] = lp_gauss
    res['d_lp'] = d_lp
    res['d_lp_fit'] = d_lp_fit
    res['slice'] = slice
    return res

def slice_thickness_summary(nifti_image, nifti_segmentation, res):
    image = nifti_image.get_fdata()
    seg1 = nifti_segmentation.get_fdata() == 1
    seg2 = nifti_segmentation.get_fdata() == 2
    head = nifti_image.header
    resolution = head.get('pixdim')[1:4]
    
    x_axis = np.arange(res['lp'].shape[1])*resolution[1]
    id1 = np.where(seg1)
    h_1 = np.arange(np.min(id1[0]), np.max(id1[0])+1)
    id2 = np.where(seg2)
    h_2 = np.arange(np.min(id2[0]), np.max(id2[0])+1)

    x0 = np.min(id1[1])
    l = len(res['lp'][0])-1

    fig = plt.figure(figsize=(14,6))
    plt.tight_layout()

    # Phantom slice
    fig.add_subplot(1,3,1)
    plt.imshow(image[:,:,res['slice']], cmap='gray'); plt.axis('off')

    # Rectangular ROI over each wedge
    lw = 0.5
    plt.plot([x0,x0+l], [h_1[0],h_1[0]], '-r', linewidth=lw)
    plt.plot([x0,x0+l], [h_1[-1],h_1[-1]], '-r', linewidth=lw)
    plt.plot([x0,x0], [h_1[0],h_1[-1]], '-r', linewidth=lw)
    plt.plot([x0+l,x0+l],[h_1[0],h_1[-1]], '-r', linewidth=lw)
    plt.plot([x0,x0+l], [h_2[0],h_2[0]], '-b', linewidth=lw)
    plt.plot([x0,x0+l], [h_2[-1],h_2[-1]], '-b', linewidth=lw)
    plt.plot([x0,x0], [h_2[0],h_2[-1]], '-b', linewidth=lw)
    plt.plot([x0+l,x0+l], [h_2[0],h_2[-1]], '-b', linewidth=lw)

    # Mean intensity along each wedge with respect to image position (x) and the fitted Gaussian curves
    fig.add_subplot(1,3,2)
    plt.plot(x_axis, res['lp'][0], 'xr', linewidth=1)
    plt.plot(x_axis, res['lp'][1], 'xb', linewidth=1)
    plt.plot(x_axis, res['lp_gauss'][0], '-or', linewidth=1)
    plt.plot(x_axis, res['lp_gauss'][1], '-ob', linewidth=1)
    plt.xlabel('Position [mm]')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.5)
    plt.axis([0,max(x_axis), 0, None])

    # Derivative of the mean intensity along each wedge with respect to image position (x)
    fig.add_subplot(1,3,3)
    plt.plot(x_axis, res['d_lp'][0], 'xr')
    plt.plot(x_axis, res['d_lp'][1], 'xb')
    plt.plot(x_axis, res['d_lp_fit'][0], '--r', linewidth=1)
    plt.plot(x_axis, res['d_lp2_fit'][1], '--b', linewidth=1)

    plt.show()