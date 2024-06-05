import os
import json

import ants
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, distance_transform_edt
from tqdm import tqdm

from .misc import ghost_path
from .metrics import calc_psnr
from .math import rician_loglike, make_sphere
from .dataio import _get_image

def _check_fname(fname):
    if os.path.exists(fname):
        return os.path.abspath(fname)
    else:
        raise FileNotFoundError(f"Can't find {fname}")

class Phantom():

    def __init__(self, phantom_name):
        self.path = os.path.join(ghost_path(), 'data', phantom_name)
        self.spec_json = os.path.join(self.path, 'spec.json')
        self.weightings = []

    def get_phantom_nii(self, weighting='T1w'):
        """Get filename of phantom image

        Args:
            weighting (str, optional): Which weighting (T1w, T2w)

        Raises:
            ValueError: Wrong weighting

        Returns:
            str: Full file path
        """

        if weighting not in self.weightings:
            raise ValueError(f'Not a valid weighting. (Valid: {self.weightings})')
        else:
            return _check_fname(os.path.join(self.path, f'phantom_{weighting}.nii.gz'))
        
        
    def reg_to_phantom(self, target_img, phantom_weighting='T1', mask='phantomDilMask'):
        """Get transformation object from target image to reference image
        
        Parameters
        ----------
        target_img : antsImage
            The target image.
        
        phantom_weighting : str
            Which weighting.

        mask : str
            Which mask to use. Should be a valid segmentation in the phantom data directory

        Returns
        -------
        list
            Inverse transforms used to warp segmentation from reference to taget space.
        """

        ref_img = ants.image_read(self.get_phantom_nii(phantom_weighting))
        mask_img = ants.image_read(self.get_seg_nii(mask))
        
        reg_rigid = ants.registration(fixed=ref_img, moving=target_img, type_of_transform='Affine')
        reg_elastics = ants.registration(fixed=ref_img, moving=target_img, type_of_transform='SyN', mask=mask_img,
                                    initial_transform=reg_rigid['fwdtransforms'][0], total_sigma=200, flow_sigma=30)
        
        return reg_elastics['invtransforms']

    def get_seg_nii(self, seg='T1'):
        """Get filename of segmentation image

        Args:
            seg (str, optional): Which segmentation (T1, T2, ADC, LC, fiducials, wedges). Default is 'T1'.

        Raises:
            ValueError: Wrong segmentation

        Returns:
            str: Full file path
        """    
        
        return _check_fname(os.path.join(self.path, f'seg_{seg}.nii.gz'))

    def warp_seg(self, target_img, seg='T1', xfm=None, weighting=None):
        """Warp any segmentation to target image
        
        Parameters
        ----------
        target_img : ANTsImage
            The reference image.
        
        seg : str
            Which segmentation to warp

        xfm : ANTsTransform
            Transformation object to use. Will calculate if not provided

        weighting : str
            Which phantom weighting to register to
        
        Returns
        -------
        ANTsImage
            The warped segmentation.
        """
        
        if xfm is None and weighting is None:
            raise ValueError('Either xfm or weighting must be provided')
        
        elif xfm is None and weighting is not None:
            xfm = self.reg_to_phantom(target_img, phantom_weighting=weighting)

        seg = ants.image_read(self.get_seg_nii(seg))
        seg_warp = ants.apply_transforms(fixed=target_img, moving=seg, 
                                        transformlist=xfm, interpolator='genericLabel')
        
        return seg_warp, xfm

    def get_specs(self):
        with open(self.spec_json, 'r') as f:
            D = json.load(f)
        return D
        
class Caliber137(Phantom):
        
    def __init__(self):
        self.name = 'Caliber137'
        super().__init__(self.name)
        self.weightings = ['T1w', 'T2w']
        
    def get_array_conc(self, array):
        D = self.get_specs()
        return D['Arrays'][array]

    def get_LC_specs(self):
        D = self.get_specs()
        return D['Other']['LC']
    
    def get_fill_specs(self):
        D = self.get_specs()
        return D['Other']['Fill']

    def get_phantom_location(self, seg):
        loc = self.get_specs()['Locations']
        return loc[seg]
    
    def loglike_temp(self, thermo, LC, plot_on=False):
        """
        Calculate the phantom temperature based on loglike estimation.

        Args:
            thermo (ndarray): The thermal image data.
            LC (ndarray): The LC image data.
            plot_on (bool, optional): Whether to plot the results. Defaults to False.

        Returns:
            float: The estimated temperature.
            Figure: The figure object if `plot_on` is True, otherwise None.
        """

        thermo = _get_image(thermo)
        LC = _get_image(LC)

        LC = LC.numpy()
        thermo = thermo.numpy()

        nvox = len(LC[LC > 0])
        x = np.zeros(nvox)
        y = np.zeros(nvox)
        idx = 0

        for i in range(1, int(np.max(LC))):
            data = thermo[LC == i]
            y[idx:idx + len(data)] = data
            x[idx:idx + len(data)] = i
            idx += len(data)

        data = thermo[LC == 11]
        y[idx:idx + len(data)] = data
        x[idx:idx + len(data)] = 0

        tri_groups = np.tri(11)
        LL = np.zeros(11)
        for i in range(11):
            means = np.zeros_like(x)
            sigmas = np.zeros_like(x)
            groups = np.array([int(tri_groups[i, int(xi)]) for xi in x])

            for k in range(0 if (i < len(tri_groups) - 1) else 1, 2):
                means[groups == k] = np.mean(y[groups == k])
                sigmas[groups == k] = np.std(y[groups == k])

            LL[i] = rician_loglike(y, sigmas, means).sum()

        LL_max = np.argmax(LL)
        temperatures = np.arange(14, 25)
        my_temperature = temperatures[LL_max]

        if plot_on:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(np.rot90(thermo[:, :, 23]), cmap='gray')
            axes[0].axis('off')
            LC_img = LC
            LC_img[LC_img == 0] = np.nan
            axes[0].imshow(np.rot90(LC_img[:, :, 23]), cmap='jet', alpha=0.5)

            axes[1].plot(x[x < LL_max + 1], y[x < LL_max + 1], '.', color='C00', alpha=0.2)
            axes[1].plot(x[x > LL_max], y[x > LL_max], '.', color='C01', alpha=0.2)
            axes[1].set_xlabel('ROI')

            for i in range(LL_max + 1):
                axes[1].plot(x[x == i][0], np.mean(y[x == i]), 'o-', color='C00', alpha=1)

            for i in range(LL_max + 1, 11):
                axes[1].plot(x[x == i][0], np.mean(y[x == i]), 'o-', color='C01', alpha=1)

            axes[1].set_ylabel('Image intensity')
            axes[1].grid('both')
            axes[1].set_xticks(np.arange(0, 11))

            axes[2].plot(temperatures, LL, 'ok-')
            axes[2].plot(temperatures[LL_max], LL[LL_max], 'ro')
            axes[2].set_xlabel('Temperature')
            axes[2].set_ylabel('log(L)')
            axes[2].set_xticks(temperatures)
            axes[2].grid('both')

            plt.tight_layout()

            return my_temperature, fig

        return my_temperature

    def get_ref_fiducial_locations(self):
        """
        Returns the reference fiducial locations.

        Returns:
            positions (numpy.ndarray): An array of shape (3, 15) containing the reference fiducial locations.
        """
        pos_D = self.get_specs()["FiducialPositions"]
        positions = np.zeros([3,15])
        for i in range(1,16):
            positions[:,i-1] = pos_D[str(i)]
        
            return positions
    
    def segment_fiducials(self, img, xfm=None, weighting='T2w', binarize_threshold=0.5, verbose=True):
        """
        Segment the distortion fiducials in an phantom image.

        Parameters:
        - img: The input image to segment fiducials from.
        - xfm: The initial affine transformation. If not provided, an initial affine registration will be performed.
        - weighting: The weighting used for registration. Default is 'T2w'.
        - binarize_threshold: The threshold used for binarizing the fiducial masks. Default is 0.5.
        - verbose: Whether to print verbose output. Default is True.

        Returns:
        - warped_fids_4D: A 4D image containing the warped fiducials.
        - mask_img_3D: A 3D image containing the segmented fiducials.
        - xfm: The final affine transformation.
        - refined_xfm: A list of refined affine transformations for each fiducial.
        """
        
        phantom_T2 = ants.image_read(self.get_phantom_nii(weighting=weighting))
        source_fid = ants.image_read(self.get_seg_nii('fiducials'))

        dx,dy,dz = phantom_T2.spacing
        fid_radius_mm = int(self.get_specs()['Sizes']['Fiducials'])/2
        fid_radius_vox = fid_radius_mm/dx

        if not xfm:
            print('Initial affine registration')
            aff = ants.registration(img, phantom_T2, type_of_transform='Affine')
            xfm = aff['fwdtransforms'][0]

        phantom_mask_reg = ants.apply_transforms(img, ants.image_read(self.get_seg_nii('phantomMask')), 
                                                transformlist=[xfm], interpolator='genericLabel')
        
        refined_xfm = []
        warp_fids = []

        for fid_id in tqdm(range(1,16), desc='Refining fiducials'):
            com = center_of_mass((source_fid==fid_id).numpy())
            sphere_arr = make_sphere(source_fid.shape, fid_radius_vox*3, com)

            sphere_mask = ants.from_numpy(sphere_arr, origin=source_fid.origin, direction=source_fid.direction, spacing=source_fid.spacing)
            sphere_mask_reg = ants.apply_transforms(img, sphere_mask, transformlist=[xfm], interpolator='genericLabel') * phantom_mask_reg
            
            # Swoop distnance image
            otsu = ants.otsu_segmentation(img, mask=sphere_mask_reg, k=1)
            otsu_arr = (1-otsu.numpy())*sphere_mask_reg.numpy()
            otsu_edt = ants.from_numpy(distance_transform_edt(otsu_arr), origin=otsu.origin, direction=otsu.direction, spacing=otsu.spacing)

            # Ref imag
            ref_edt = ants.from_numpy(distance_transform_edt((source_fid==fid_id).numpy()), origin=source_fid.origin, 
                                    direction=source_fid.direction, spacing=source_fid.spacing)
            
            reg = ants.registration(fixed=otsu_edt, 
                                moving=ref_edt, 
                                initial_transform=xfm, 
                                type_of_transform='Rigid',
                                mask=sphere_mask_reg,
                                mask_all_stages=True,
                                aff_metric='meansquares', 
                                aff_random_sampling_rate=1,
                                aff_iterations=(2000),
                                aff_shrink_factors=(1),
                                aff_smoothing_sigmas=(0))
            
            refined_xfm.append(reg['fwdtransforms'][0])
            
            fid_reg = ants.apply_transforms(img, (source_fid == fid_id), transformlist=reg['fwdtransforms'], interpolator='linear')
            warp_fids.append(fid_reg)

        mask_data_4D = np.zeros([*warp_fids[0].shape, 15])
        mask_data_3D = np.zeros(warp_fids[0].shape)
        
        for i in range(15):
            mask_data_4D[...,i] = warp_fids[i].to_nibabel().get_fdata()
            mask_data_3D += np.where(warp_fids[i].numpy() > binarize_threshold, 1, 0) * (i+1)

        warped_fids_4D = ants.from_nibabel(nib.Nifti1Image(mask_data_4D, affine=warp_fids[0].to_nibabel().affine))
        mask_img_3D = ants.from_numpy(mask_data_3D, origin=img.origin, spacing=img.spacing, direction=img.direction)

        return warped_fids_4D, mask_img_3D, xfm, refined_xfm


    def find_fiducials_old(self, swoop_img, xfm=None, weighting='T2w', verbose=True):
        phantom_T2 = ants.image_read(self.get_phantom_nii(weighting))
        
        dx,dy,dz = phantom_T2.spacing
        fid_radius_mm = int(self.get_specs()['Sizes']['Fiducials'])/2
        fid_radius_vox = fid_radius_mm/dx
        
        if not xfm:
            print("Affine registration to template")
            dense_rigid = ants.registration(swoop_img, phantom_T2, type_of_transform='DenseRigid')
            aff = ants.registration(swoop_img, phantom_T2, type_of_transform='Affine', initial_transform=dense_rigid['fwdtransforms'][0])
            xfm = aff['fwdtransforms'][0]

        source_fid = ants.image_read(self.get_seg_nii('fiducials'))
        warp_fids = ants.apply_transforms(swoop_img, source_fid, transformlist=[xfm], interpolator='genericLabel')
        
        images_out = []
        masks_out = []
        refined_xfm = []

        for i in tqdm(range(1,16), desc='Refining segmentation', disable=(not verbose)):
            single_fid = (source_fid==i) * (-1.0) + 1

            fid_guess = (warp_fids==i).numpy()
            sphere = make_sphere(warp_fids.shape, fid_radius_vox*3, center_of_mass(fid_guess))
            single_mask = ants.from_numpy(sphere, origin=warp_fids.origin, spacing=warp_fids.spacing, direction=warp_fids.direction)

            reg = ants.registration(swoop_img, single_fid, 
                                    initial_transform=xfm, 
                                    type_of_transform='Affine', 
                                    mask=single_mask, aff_metric='mattes', 
                                    aff_random_sampling_rate=1,
                                    aff_iterations=(2000,2000),
                                    aff_shrink_factors=(2,1),
                                    aff_smoothing_sigmas=(1,0))
            
            refined_xfm.append(reg['fwdtransforms'][0])
            
            images_out.append(((reg['warpedmovout']* (-1.0) + 1)*single_mask).to_nibabel())
            masks_out.append(single_mask.to_nibabel())

        affine = images_out[0].affine
        new_data = np.zeros((*images_out[0].shape,15))
        new_data2 = np.zeros((*images_out[0].shape,15))

        for i in range(15):
            new_data[...,i] = images_out[i].get_fdata()
            new_data2[...,i] = masks_out[i].get_fdata()

        new_img = nib.Nifti1Image(new_data, affine=affine)
        new_img2 = nib.Nifti1Image(new_data2, affine=affine)
        
        return ants.from_nibabel(new_img), ants.from_nibabel(new_img2), xfm, refined_xfm

