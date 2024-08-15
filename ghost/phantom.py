import json
import os
import tempfile

import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt
from skimage.draw import disk
from tqdm import tqdm

from .dataio import _get_image
from .misc import ghost_path
from .reg import exhaustive_initializer
from .utils import make_sphere, rician_loglike


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
        self.valid_seg = []


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


    def robust_initializer(self, target_img, weighting='T2w', mask='phantomMask'):
        """
        Wrapper for robust initialization of z rotation

        Args:
            target_img (numpy.ndarray): The target image to be registered.
            weighting (str, optional): Phantom weighting. Defaults to 'T2w'.
            mask (str, optional): The mask used in phantom space. Defaults to 'phantomMask'.

        Returns:
            ants.core.transforms.Transform: transformation

        """

        fixed_fname = self.get_phantom_nii(weighting)
        mask_fname = self.get_seg_nii(mask)

        with tempfile.TemporaryDirectory() as td:
            print("Exhaustive initializer")
            moving_fname = os.path.join(td, 'moving.nii.gz')
            ants.image_write(target_img, moving_fname)

            xfm_ants = exhaustive_initializer(fixed_fname=fixed_fname, moving_fname=moving_fname, mask_fname=mask_fname)
    
            return xfm_ants


    def reg_to_phantom(self, target_img, do_syn=True, syn_total_sigma=200, syn_flow_sigma=30, weighting='T1', mask='phantomMask', init_z=True):

        """Get transformation object from target image to reference image
        
        Parameters
        ----------
        target_img : antsImage
            The target image.
        
        do_syn : bool
            To run SyN (deformable) registration
            
        syn_total_sigma : int
            Total sigma for SyN registration
        
        syn_flow_sigma : int
            Flow sigma for SyN Registration

        weighting : str
            Which weighting.

        mask : str
            Which mask to use. Should be a valid segmentation in the phantom data directory

        Returns
        -------
        (list,list)
            Filenames of inverse and forward transforms
        """
        
        ref_img = ants.image_read(self.get_phantom_nii(weighting))
        mask_img = ants.image_read(self.get_seg_nii(mask))

        init_xfm = None

        if init_z:
            xfm = self.robust_initializer(target_img, weighting=weighting, mask=mask)
            xfm_fname = tempfile.mkstemp(suffix='.mat')[1]
            ants.write_transform(xfm, xfm_fname)
            
            init_xfm = xfm_fname
 
        reg = ants.registration(fixed=ref_img, moving=target_img, mask=mask_img, type_of_transform='DenseRigid', initial_transform=init_xfm)

        if do_syn:
            reg = ants.registration(fixed=ref_img, moving=target_img, type_of_transform='SyN', mask=mask_img,
                                        initial_transform=reg['fwdtransforms'][0], total_sigma=syn_total_sigma, flow_sigma=syn_flow_sigma)
        
        return reg['invtransforms'], reg['fwdtransforms']


    def get_seg_nii(self, seg='T1'):
        """Get filename of segmentation image

        Args:
            seg (str, optional): Which segmentation. Default is 'T1'.

        Raises:
            ValueError: Wrong segmentation

        Returns:
            str: Full file path
        """    
        
        return _check_fname(os.path.join(self.path, f'seg_{seg}.nii.gz'))


    def warp_seg(self, target_img, xfm, seg):
        """Warp any segmentation to target image
        
        Parameters
        ----------
        target_img : ANTsImage
            The reference image.
        
        seg : str
            Name of the segmentation to warp

        xfm : xfm
            The transforms to apply    
    
        Returns
        -------
        ANTsImage
            The warped segmentation.
        """

        if xfm is None:
            raise ValueError('xfm must be provided. Run reg_to_phantom first')
        
        if type(xfm) == str:
            xfm = [xfm]
        
        if len(xfm)>1:
            whichtoinvert = [True, False]
        else:
            whichtoinvert = [True]


        seg_warp = ants.apply_transforms(fixed=target_img, 
                                         moving=ants.image_read(self.get_seg_nii(seg)), 
                                         transformlist=xfm, interpolator='genericLabel', whichtoinvert=whichtoinvert)
        
        return seg_warp


    def get_specs(self):
        with open(self.spec_json, 'r') as f:
            D = json.load(f)
        return D


class Caliber137(Phantom):
        
    def __init__(self):
        self.name = 'Caliber137'
        super().__init__(self.name)
        self.weightings = ['T1w', 'T2w']
        self.valid_seg = ['ADC', 'BG', 'fiducials', 'LC', 'phantomMask', 
                          'T1mimics', 'T2mimics', 'wedges']
        
    def get_array_conc(self, array):
        D = self.get_specs()
        
        if array=='ADC':
            conc = {0:2393, 10:1884, 20:1439, 30:1047, 40:654, 50:388}
            ADC_vals = [conc[x] for x in D['Arrays'][array]['Concentration']]
            D['Arrays'][array]['ADC'] = ADC_vals

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
            plt.style.use('default')
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

    def get_seg_z_location(self, img_affine, seg_name, xfm_fname, offset=0):
        tx = ants.read_transform(xfm_fname)
        z = self.get_phantom_location(seg_name) + offset
        p = [*tx.apply_to_point([0,0,z]), 1]
        ijk = np.linalg.inv(img_affine) @ p
        
        return ijk, p[:3]

    def mimic_3D_to_2D_axial(self, seg_img, seg_name, xfm_fname, radius=None):
        dx,dy,dz = seg_img.spacing
        
        my_k = 0
        my_r = 0
        
        if not radius:
            radius = int(self.get_specs()['Sizes'][seg_name])/2

        mimic_radius_vox = int(radius)/dx
        
        img_affine = seg_img.to_nibabel().affine
        ijk, xyz_ref = self.get_seg_z_location(img_affine, seg_name, xfm_fname, offset=0)
        z0 = xyz_ref[2]

        klist = [int(np.floor(ijk[2])), int(np.ceil(ijk[2]))]
        
        for k0 in klist:
            # Figure out which z position we are at relative to center of the sphere
            z1 = (img_affine @ np.array([0,0,k0,1]))[2]
            
            if mimic_radius_vox**2 > (z0 - z1)**2:
                pv_rad = np.sqrt(mimic_radius_vox**2 - (z0 - z1)**2)
            else:
                pv_rad = 0
            
            if pv_rad > my_r:
                my_k = k0
                my_r = pv_rad
        
        new_seg = np.zeros(seg_img.shape[0:2])

        for i in range(1,15):
            com = center_of_mass(seg_img.numpy()==i)
            d = disk([com[0], com[1]], my_r, shape=seg_img.shape[0:2])
            new_seg[d] = i

        refined_seg = np.zeros(seg_img.shape)
        refined_seg[...,my_k] = new_seg
        refined_seg_img = ants.from_numpy(refined_seg, origin=seg_img.origin, direction=seg_img.direction, spacing=seg_img.spacing)
        
        return refined_seg_img

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

    