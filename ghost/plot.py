"""    
For all functions that does any plotting or visualisation
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_slice_thickness_distribution(plot_data, target_img):
    """
    Plots the slice thickness distribution for a given slice.
    
    Parameters
    ----------
    plot_data : dict
        A dictionary of variables needed to plot the slice thickness distribution.

    target_img : ANTsImage
        The target image (axi). Probably from the Swoop.
    """
    # Unpack dictionary
    for key, value in plot_data.items():
        locals()[key] = value

    seg1 = wseg == 1
    seg2 = wseg == 2

    id1 = np.where(seg1)
    h_1 = np.arange(np.min(id1[0]), np.max(id1[0])+1)
    id2 = np.where(seg2)
    h_2 = np.arange(np.min(id2[0]), np.max(id2[0])+1)

    x0 =  np.min(id1[1])
    l = lp.shape[1]-1

    fig = plt.figure(figsize=(14,6))
    plt.tight_layout()

    #
    # SUBPLOT 1
    #
    fig.add_subplot(1,3,1)

    # Plot phantom slice
    plt.imshow(my_swoop_img[:,:,slice], cmap='gray'); plt.axis('off')

    # Plot rectangular ROI over each wedge
    lw = 0.5
    plt.plot([x0,x0+l], [h_1[0],h_1[0]], '-r', linewidth=lw)
    plt.plot([x0,x0+l], [h_1[-1],h_1[-1]], '-r', linewidth=lw)
    plt.plot([x0,x0], [h_1[0],h_1[-1]], '-r', linewidth=lw)
    plt.plot([x0+l,x0+l],[h_1[0],h_1[-1]], '-r', linewidth=lw)
    plt.plot([x0,x0+l], [h_2[0],h_2[0]], '-b', linewidth=lw)
    plt.plot([x0,x0+l], [h_2[-1],h_2[-1]], '-b', linewidth=lw)
    plt.plot([x0,x0], [h_2[0],h_2[-1]], '-b', linewidth=lw)
    plt.plot([x0+l,x0+l], [h_2[0],h_2[-1]], '-b', linewidth=lw)

    #
    # SUBPLOT 2
    #
    # In subplot 2, the plot shows the mean intensity along each wedge with respect to image position (x) and the fitted Gaussian curves
    fig.add_subplot(1,3,2)
    plt.plot(x, lp[0], 'xr', linewidth=1)
    plt.plot(x, lp[1], 'xb', linewidth=1)
    plt.plot(x, lp_smooth[0], '-or', linewidth=1)
    plt.plot(x, lp_smooth[1], '-ob', linewidth=1)
    plt.xlabel('Position [mm]')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.5)
    plt.axis([0,max(x), 0, None])

    #
    # SUBPLOT 3
    #
    # In subplot 3, the plot shows the derivative of the mean intensity along each wedge with respect to image position (x), fitted to a Gaussian distribution.
    fig.add_subplot(1,3,3)
    plt.plot(x, lp_diff[0], 'xr')
    plt.plot(x, lp_diff[1], 'xb')
    #plt.plot(x,  [0], '--r', linewidth=1)
    #plt.plot(x, lp_fit[1], '--b', linewidth=1)

    plt.show()