import os
import bids
import ants
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from ghost.stats import *
from ghost.calib import *
from ghost.phantom import *

def get_mask_nii(seg, target_BIDSImageFile, layout, verbose=False):
    """ Retrieve or create specified segmentation mask for target image. 
    All other functions depend on it since it is the smallest building block. 
    Perhaps it is useful as a standalone function for the cheeky scientist.
    
    Parameters
    ----------
    seg : str
        Which segmentation to use (T1, T2, ADC, fiducials, wedges).

    target_BIDSImageFile : BIDSImageFile
        The BIDSImageFile object to which the segmentation will be warped.

    layout : BIDSLayout
        The BIDSLayout object.

    verbose : bool
        Whether to print the status of the function. Default is False.

    Returns
    -------
    mask_path : path
        Path to the warped segmentation.

    Example
    -------
    >>> project_dir = 'path/to/project/directory/that/contains/rawdata/and/derivatives'
    >>> layout = bids.BIDSLayout(root=project_dir+'/rawdata', derivatives=project_dir+'/derivatives')
    >>> bids_files = layout.get(scope='raw', extension='.nii.gz', suffix='dwi', acquisition='adc')
    >>> for bf in bids_files:
    >>>     mask_path = get_mask_nii('ADC', bids_files[0], layout)
    >>>     some_custom_function(mask_path, bf.path)
    """
    def mask_exists(target_img, seg, layout):
        """Check if a seg exists in the derivatives tree for a given bids image."""
        target_entities = target_img.get_entities() # Get the entities for the target image
        mask_entities = target_entities.copy() # Get the entities for the mask
        mask_entities["desc"] = seg # Add the desc entity to the mask entities
        return layout.get(scope='derivatives', **mask_entities)

    # Get filename of mask
    mask_name = target_BIDSImageFile.filename[:-10] + 'desc-' + seg + '_' + target_BIDSImageFile.entities['suffix'] + '.nii.gz'
    mask_dir = layout.get(scope='derivatives')[0].dirname + '/masks/sub-' + target_BIDSImageFile.entities['subject'] + '/ses-' + target_BIDSImageFile.entities['session'] + '/' + target_BIDSImageFile.entities['datatype']
    mask_path = mask_dir + '/' + mask_name

    if not os.path.exists(mask_dir): os.makedirs(mask_dir)

    # Find out if mask already exists
    if mask_exists(target_BIDSImageFile, seg, layout):
        if verbose: print(f"A {seg}-mask already exists for {target_BIDSImageFile.filename}.")
        
        return mask_path
    else:
        if verbose: print(f"A {seg}-mask does not exist for {target_BIDSImageFile.filename}. Molding one now.")
        
        suffix = target_BIDSImageFile.entities['suffix']
        if suffix == 'T1w':
            ref = 'T1'
        elif suffix == 'T2w' or suffix == 'FLAIR' or suffix == 'dwi':
            ref = 'T2'
        else: raise ValueError("The image type is not supported.")

        # If the target image is ADC or b=900, use the b=0 image as the reference
        # Otherwise, use the target image as the reference
        acq = target_BIDSImageFile.entities['acquisition']
        if acq == 'adc' or acq == 'b900':
            b0_entities = target_BIDSImageFile.get_entities()
            b0_entities['acquisition'] = 'b0'
            target_ANTsImage = ants.image_read(layout.get(scope='raw', **b0_entities)[0].path, reorient=True)
        
        else: 
            target_ANTsImage = ants.image_read(target_BIDSImageFile.path)

        mask = warp_seg(target_img=target_ANTsImage, weighting=ref, seg=seg)

        if verbose: print(f"Saving {seg} mask to {mask_path}...")
        ants.image_write(mask, mask_path)

        return mask_path

def bids2stats(target, seg, layout, toExcel=False, verbose=False):
    """ Get the stats of a specific segmentation for a list of BIDS files
    
    Parameters
    ----------
    target : list of BIDSImageFiles
        The list containing all the BIDSImageFiles to get the stats from. Can be a single file.
    
    seg : str
        Segmentation(s) to use (T1, T2, ADC, LC, fiducials, wedges). May also be a path to your own mask, just make sure the mask is placed in the /derivatives/masks/ folder.

    layout : BIDSLayout
        The BIDS layout to use.

    Returns
    -------
    stats : pandas dataframe
        A dataframe with LabelValue, mean, min, max, variance, count, volume, session, acquisition, orientation, modality, run, segmentation.
    
    Example
    -------
    >>> project_dir = 'path/to/project/directory/that/contains/rawdata/and/derivatives'
    >>> layout = bids.BIDSLayout(root=project_dir+'/rawdata', derivatives=project_dir+'/derivatives')
    >>> bids_files = layout.get(scope='raw', extension='.nii.gz', suffix='dwi', acquisition='adc')
    >>> stats = bids2stats(bids_files, 'ADC', layout, toExcel=False)
    """
    possible_seg = ['T1', 'T2', 'ADC', 'LC', 'fiducials', 'wedges']
    stats_merged = pd.DataFrame()

    for target_bf in target: # bf = BIDSFile
        if seg in possible_seg:
            mask_path = get_mask_nii(seg, target_bf, layout, verbose=verbose)
        else: 
            mask_path = seg
            if verbose: print("Using custom mask.")
        mask_img = ants.image_read(mask_path)
        target_img = ants.image_read(target_bf.path)
        
        # Use parse_rois to get the stats and append the following entities
        stats = parse_rois(target_img, mask_img)
        stats['Session'] = target_bf.entities['session']
        stats['Acquisition'] = target_bf.entities['acquisition']
        stats['Modality'] = target_bf.entities['suffix']
        if seg in possible_seg: 
            stats['Segmentation'] = seg
        else: 
            stats['Segmentation'] = 'custom'

        if 'run' not in target_bf.entities: 
            stats['Run'] = 'NA'
        else: 
            stats['Run'] = target_bf.entities['run']

        if 'reconstruction' not in target_bf.entities: 
            stats['Orientation'] = 'NA'
        else: 
            stats['Orientation'] = target_bf.entities['reconstruction']

        if stats_merged.empty: 
            stats_merged = stats
        else: 
            stats_merged = pd.concat([stats_merged, stats], ignore_index=True)

    stats_merged = stats_merged.sort_values(by=['Session', 'Segmentation', 'LabelValue', 'Run', 'Acquisition', 'Orientation', 'Modality'])

    if toExcel: # save stats to the derivatives folder as an excel file
        acq_labels = '_'.join(stats_merged['Acquisition'].unique())
        rec_labels = '_'.join(stats_merged['Orientation'].unique())
        mod_labels = '_'.join(stats_merged['Modality'].unique())
        seg_label = str(seg if seg in possible_seg else 'custom')
        # Get filename of stats file
        stats_name = 'acq-' + acq_labels + '__rec-' + rec_labels + '__desc-' + seg_label + '__' + mod_labels + '.xlsx'
        # print(f"Stats name: {stats_name}")
        stats_dir =  mask_path.split("/masks/")[0] + '/stats'
        # print(f"Stats directory: {stats_dir}")
        stats_path = stats_dir + '/' + stats_name

        # Create the stats directory if it doesn't exist
        if not os.path.exists(stats_dir): os.makedirs(stats_dir)

        # Save the stats to an excel file
        stats_merged.to_excel(stats_path, index=False)

    return stats_merged

def plot_mimics(target, layout, toFile=True, verbose=False):
    """ Plot the T2, ADC, and T1 ROIs for a list of BIDS files
    
    Parameters
    ----------
    target : list of BIDSImageFiles
        The list containing all the BIDSImageFiles to get the stats from. Can be a single file.

    layout : BIDSLayout
        The BIDS layout to use.

    toFile : bool, optional (default: True)
        If True, the plot will be saved to a png file.

    Returns
    -------
    None

    Example
    -------
    >>> project_dir = 'path/to/project/directory/that/contains/rawdata/and/derivatives'
    >>> layout = bids.BIDSLayout(root=project_dir+'/rawdata', derivatives=project_dir+'/derivatives')
    >>> bids_files = layout.get(scope='raw', extension='.nii.gz', suffix='T1w') # Get all T1 and T2 images
    >>> bids_files = bids_files[:3] # Only use the first 3 files to save some time
    >>> plot_mimics(bids_files, layout) # Boom! You have a plot!
    """
    labels = ['T2', 'ADC', 'T1']
    masks = {}
    for i, target_bf in enumerate(target):
        for l in labels:
            mask_path = get_mask_nii(l, target_bf, layout, verbose)
            masks[l] = ants.image_read(mask_path, reorient=True).numpy()
        target_img = ants.image_read(target_bf.path, reorient=True)

        fig = plt.figure(figsize=(8, 3))
        plt.style.use("dark_background")
        cmaps = ['Reds', 'Blues', 'Greens']
        for i, l in enumerate(labels):
            masks[l][masks[l] == 0] = np.nan
            fig.add_subplot(1,3,i+1)
            # Check acquisition and reconstruction to determine which slices to plot
            acq = target_bf.entities['acquisition']
            suf = target_bf.entities['suffix']
            rec = target_bf.entities['reconstruction']
            
            if acq == 'adc': # suf == 'dwi' and rec == axi, always
                slices = [12, 17, 23]
            elif rec == 'axi':
                    slices = [14, 18, 25]
            # rec == 'cor' or rec == 'sag'
            elif acq == 'gw' or acq == 'fast':
                    slices = [43, 58, 79]                            
            elif acq == 'std':
                    if suf == 'FLAIR':
                        slices = [40, 55, 74]
                    elif suf == 'T1w':
                        slices = [43, 58, 79]
                    elif suf == 'T2w':
                        if rec == 'cor':
                            slices = [46, 62, 84]
                        elif rec == 'sag':
                            slices = [52, 68, 90]

            plt.imshow(target_img[:,:,slices[i]], cmap='gray', aspect=target_img.spacing[0]/target_img.spacing[1])
            plt.imshow(masks[l][:,:,slices[i]], cmap=cmaps[i], alpha=0.5, aspect=target_img.spacing[0]/target_img.spacing[1])
            plt.title(l)
            plt.axis('off')
        suptitle = fig.suptitle(target_bf.filename[:-7], fontsize=12)
        plt.tight_layout()

        if not toFile:
            pass
        
        # Save the figure to derivatives/images
        img_name = target_bf.filename[:-10] + 'desc-mimics' + '_' + target_bf.entities['suffix'] + '.png'
        img_dir = layout.get(scope='derivatives')[0].dirname + '/qa/sub-' + target_bf.entities['subject'] + '/ses-' + target_bf.entities['session'] + '/' + target_bf.entities['datatype']
        img_path = img_dir + '/' + img_name

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        plt.savefig(img_path)

def plot_signal_ROI(target, seg, layout, verbose=False):
    """ Plot the signal in different ROIs...
    
    Parameters
    ----------
    target : list of BIDSImageFiles
        A list containing all the BIDSImageFiles from different dates to get the stats from.

    seg : str
        The name of the segmentation to use.

    layout : BIDSLayout
        The BIDS layout to use.

    Returns
    -------
    None

    Example
    -------
    >>> project_dir = 'path/to/project/directory/that/contains/rawdata/and/derivatives'
    >>> layout = bids.BIDSLayout(root=project_dir+'/rawdata', derivatives=project_dir+'/derivatives')
    >>> bids_files = layout.get(scope='raw', extension='.nii.gz', suffix='dwi', acquisition='adc')
    >>> plot_signal_ROI(bids_files, 'ADC', layout)
    """

    def convert_date(date_str): # Convert Session label from YYYYMMDD to DD/MM
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        return date_obj.strftime('%d/%m')

    # Get the stats for the target files and specified segmentation
    stats = bids2stats(target, seg, layout, toExcel=False, verbose=verbose)

    # If the 'Run' column in stats contains more than 'NA' values, group 'em together.
    if len(stats[stats['Run'] != 'NA']) > 0:
        # group by 'Session' and 'LabelValue' and compute the mean and standard deviation
        stats_grouped = stats.groupby(['Session', 'LabelValue', 'Acquisition'])
        stats_mean = stats_grouped.agg({'Mean': 'mean'})
        stats_mean['Std'] = stats_grouped['Mean'].std()
        stats_mean['Run'] = '1,2,3'
        stats = stats_mean.reset_index()

    fig = plt.figure(figsize=(10, 12))
    fig.add_subplot(2, 1, 1)
    for s in stats['Session'].unique():
        df = stats[stats['Session'] == s]
        plt.plot(df['LabelValue'], df['Mean'], label=convert_date(s))

        if df['Acquisition'].item == 'adc':
            plt.fill_between(df['LabelValue'], df['Mean'] - df['Std'], df['Mean'] + df['Std'], alpha=0.33)

    plt.xlabel('Mimic')
    plt.xticks(np.arange(1, 15, 1))
    plt.xlim([1, 14])
    plt.ylabel('Mean Intensity')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.add_subplot(2, 1, 2)
    for lv in stats['LabelValue'].unique():
        df = stats[stats['LabelValue'] == lv]
        df['Session'] = df['Session'].apply(convert_date)
        plt.plot(df['Session'], df['Mean'], label=int(lv))

        if df['Acquisition'].item == 'adc':
            plt.fill_between(df['Session'], df['Mean'] - df['Std'], df['Mean'] + df['Std'], alpha=0.33)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Date')
    plt.ylabel('Mean Intensity')
    plt.xlim([np.min(df['Session']), np.max(df['Session'])])
    fig.suptitle(f"{seg}-mimics", fontsize=16)
    plt.tight_layout()
    plt.style.use("dark_background")