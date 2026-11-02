import numpy as np
from numpy.linalg import inv

def CC_moresco():
    """
    Cosmic chronometers (CC) dataset loader and covariance estimator.
    Based on Moresco et al. 2018, 2020.
    """

    # Load cosmic chronometer data
    H_file = 'Data/HzTable_MM_BC03.dat'
    z, Hz, errHz = np.genfromtxt(H_file, comments='#', usecols=(0, 1, 2), unpack=True, delimiter=',')
    ref = np.genfromtxt(H_file, comments='#', usecols=(3), unpack=True, dtype=str, delimiter=',')

    # Load model dependence (IMF, SPS, SLIB corrections)
    MPB_file = 'Data/data_MM20.dat'
    zmod, imf, slib, sps, spsooo = np.genfromtxt(MPB_file, comments='#', usecols=(0, 1, 2, 3, 4), unpack=True)

    # Interpolate fractional systematics
    imf_intp = np.interp(z, zmod, imf) / 100.0
    slib_intp = np.interp(z, zmod, slib) / 100.0
    sps_intp = np.interp(z, zmod, sps) / 100.0
    spsooo_intp = np.interp(z, zmod, spsooo) / 100.0

    # Build diagonal covariance from errors
    cov_diag = np.diag(errHz**2)

    # Systematic covariances (vectorized outer products)
    cov_imf = np.outer(Hz * imf_intp, Hz * imf_intp)
    cov_slib = np.outer(Hz * slib_intp, Hz * slib_intp)
    cov_sps = np.outer(Hz * sps_intp, Hz * sps_intp)
    cov_spsooo = np.outer(Hz * spsooo_intp, Hz * spsooo_intp)

    # Full covariance (as per Moresco 2020)
   # cov_mat = cov_diag + cov_imf + cov_slib + cov_sps + cov_spsooo
    cov_mat = cov_diag + cov_imf + cov_spsooo
    inv_cov_mat = inv(cov_mat)

    return z, Hz, errHz, zmod, imf, slib, sps, spsooo, cov_mat, inv_cov_mat
