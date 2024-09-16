import ants
import numpy as np

def rotmat(rot_angles):
    alpha = rot_angles[2]
    beta = rot_angles[1]
    gamma = rot_angles[0]

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)

    R = np.array([[ca*cb, ca*sb*sg-sa*cg, ca*sb*cg+sa*sg],
                  [sa*cb, sa*sb*sg+ca*cg, cg*sb*sb-sg*ca],
                  [-sb,   sg*cb,          cb*cg]])

    return R

def simulate_image(img, SNR, resolution, rot=[0,0,0], trans=[0,0,0]):    
    aff = ants.ANTsTransform()
    new_par = [*rotmat(rot).flatten(), *trans]
    aff.set_parameters(new_par)

    img2 = ants.apply_ants_transform_to_image(aff, img, img, 'linear')
    img3 = ants.resample_image(img2, resample_params=resolution, use_voxels=False, interp_type=4)
    img3 /= img3.max()

    sigma = 1/SNR
    noise_image = ants.add_noise_to_image(img3, 'additivegaussian', (0.0, sigma))
    noise_image2 = ants.add_noise_to_image(img3*0, 'additivegaussian', (0.0, sigma))
    return (noise_image**2 + noise_image2**2)**0.5, aff

def transform_mask(mask, resolution, aff):
    mask2 = ants.apply_ants_transform_to_image(aff, mask, mask, 'nearestneighbor')
    mask3 = ants.resample_image(mask2, resample_params=resolution, use_voxels=False, interp_type=1)
    return mask3