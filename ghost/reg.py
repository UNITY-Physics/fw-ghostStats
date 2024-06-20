import SimpleITK as sitk
import ants
from math import pi
import tempfile

def exhaustive_initializer(fixed_fname, moving_fname, shrink=[4], sigmas=[2], mattes_sampling=0.5, mattes_bins=50, samples=40, mask_fname=None):
    
    """
    Performs exhaustive initialization for z rotation for image registration

    Parameters:
    - fixed_fname (str): File path of the fixed image.
    - moving_fname (str): File path of the moving image.
    - shrink (list, optional): List of shrink factors per level. Default is [4].
    - sigmas (list, optional): List of smoothing sigmas per level. Default is [2].
    - mattes_sampling (float, optional): Percentage of samples used for computing the Mattes mutual information metric. Default is 0.5.
    - mattes_bins (int, optional): Number of histogram bins used for computing the Mattes mutual information metric. Default is 50.
    - samples (int, optional): Number of samples used for the exhaustive optimization. Default is 40.
    - mask_fname (str, optional): File path of the mask image. Default is None.

    Returns:
    - ants.core.transforms.Transform: transformation

    """
    
    fixed_image = sitk.ReadImage(fixed_fname, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_fname, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetShrinkFactorsPerLevel(shrinkFactors=shrink)
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=sigmas)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    if mask_fname:
        mask = sitk.ReadImage(mask_fname, sitk.sitkUInt8)
        R.SetMetricFixedMask(mask)
    
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=mattes_bins)
    R.SetMetricSamplingPercentage(mattes_sampling)

    tx = sitk.Euler3DTransform()
    R.SetOptimizerAsExhaustive([0,0,samples,0,0,0,])
    R.SetOptimizerScales([1, 1, 2.0 * pi/samples, 1.0, 1.0, 1.0,])

    tx = sitk.CenteredTransformInitializer(fixed_image, moving_image, tx)
    R.SetInitialTransform(tx)
    
    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(fixed_image, moving_image)

    with tempfile.TemporaryDirectory() as d:
        outTx.WriteTransform(d+'/xfm.mat')
        xfm = ants.read_transform(d+'/xfm.mat')

    return xfm