import ipdb

import numpy as np
import scipy.stats as stats


def rft_correct(z_scores, pixdim=[2, 2, 2], fwhm=[8, 8, 8], alpha=0.05):
    """Random field correction of gauss smoothed data.

    Parameters
    ----------
    stats_values: np.ndarray
        Z-scores array

    pixdim: array/list of ints or floats
        Size in milimeters of the voxel, e.g., [2, 2, 2] if the voxel size is 2mm x 2mm x 2mm.

    fwhm: array/list of ints or floats
        Size in milimiters of the FWHM smooothing kernel.

    alpha: float
        False positive rate threshold for correction.

    Returns
    -------
    p-values volume
    """
    # Non-zero voxels
    num_nonzero_voxels = np.count_nonzero(z_scores)

    # Voxel volume mm
    voxel_volume = np.prod(pixdim)

    # Tamaño del kernel mm
    resel_volume = np.prod(fwhm)

    # Calculate number of resels
    n_resels = (num_nonzero_voxels * voxel_volume)/resel_volume

    # The inverse normal CDF
    normal_dist = stats.norm()
    z_opt = normal_dist.ppf(1 - (alpha / n_resels))

    # Apply RFT statistical correction
    pos_corr = z_scores > z_opt
    neg_corr = z_scores < -z_opt

    return z_scores * (pos_corr+neg_corr)
