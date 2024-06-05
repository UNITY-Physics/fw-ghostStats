import ants
import pandas

def parse_rois(target_img, seg_img):
    """ Parses the ROI values from a segmentation image

    Parameters
    ----------
    target_img : ANTsImage
        The target image (axi). Probably from the Swoop.
    
    seg_img : ANTsImage
        The segmentation image (T1, T2, ADC, LC, fiducials, wedges).
    
    Returns
    -------
    stats : pandas dataframe
        A dataframe with LabelValue, mean, min, max, variance, count, volume.
    """
    # Get the ROI values
    stats = ants.label_stats(target_img, seg_img)
    stats = stats.drop(stats[stats['LabelValue'] == 0].index)
    stats = stats.drop(columns=['Mass', 'x', 'y', 'z', 't'])
 
    return stats