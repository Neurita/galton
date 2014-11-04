
import numpy as np
import scipy.stats as stats


def rft_correct(image_data, pixdim=[2, 2, 2], kernel=[8, 8, 8], alpha=0.05):
    """Random field correction of gauss smoothed data.

    Parameters
    ----------
    image_data: np.ndarray
        Volume data array

    pixdim: array/list of ints or floats
        Size in milimeters of the voxel, e.g., [2, 2, 2] if the voxel size is 2mm x 2mm x 2mm.

    kernel: array/list of ints or floats
        Size in milimiters of the FWHM smooothing kernel.

    alpha: float
        False positive rate threshold for correction.

    Returns
    -------
    p-values volume
    """
    # Non-zero voxels
    num_nonzero_voxels = np.count_nonzero(image_data)

    # Voxel volume mm
    voxel_volume = np.prod(pixdim)

    # Tamaño del kernel mm
    kernel_volume = np.prod(kernel)

    # Calculate number of resels
    n_resels = (num_nonzero_voxels * voxel_volume)/kernel_volume

    # The inverse normal CDF
    z_opt = normal_dist.ppf(1 - (alpha / n_resels))

    # Apply RFT statistical correction
    pos_corr = image_data > z_opt
    neg_corr = image_data < -z_opt

    return image_data * (pos_corr+neg_corr)
