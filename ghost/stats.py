"""
Everything that has to do with statistics
"""
import ants
from ghost.phantom import *

def parse_rois(target_img, weighting, seg):
    """ Parses the ROI values from a segmentation image

    Parameters
    ----------
    target_img : ANTsImage
        The target image (axi). Probably from the Swoop.
    
    seg : str
        Which segmentation to use (T1, T2, ADC, LC, fiducials, wedges). Default is 'T1'.
    
    Returns
    -------
    stats : pandas dataframe
        A dataframe with LabelValue, mean, min, max, variance, count, volume.
    """
    # Get the segmentation image
    seg_img = warp_seg(target_img=target_img, weighting=weighting, seg=seg)
    
    # Get the ROI values
    stats = ants.label_stats(target_img, seg_img)
    stats = stats.drop(stats[stats['LabelValue'] == 0].index)
    stats = stats.drop(columns=['Mass', 'x', 'y', 'z', 't'])
 
    return stats