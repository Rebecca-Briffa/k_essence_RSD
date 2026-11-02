from util import *
import scipy.linalg as la

def Pantheon_data():
    # obtain the covariance matrix for the Pantheon file
    Data_file = read_light_curve_parameters('Data/Pantheon+SH0ES.dat')
    default_data_file = 'Data/' + 'Pantheon+SH0ES.dat'
    default_covmat_file = 'Data/'  + 'Pantheon+SH0ES_STAT+SYS.cov'

    filename = default_data_file
    #print("Loading data from {}".format(filename))
    #data = pd.read_csv(filename, delim_whitespace=True)
    light_curve_params = Data_file
    data = light_curve_params
    origlen = len(data)

    # Hubble-flow cut, with every Cepheid calibrator kept (CosmoSIS Pantheon+SH0ES).
    ww = (data['zHD'] > 0.01) | (np.asarray(data['IS_CALIBRATOR']).astype(bool))


    zCMB = data['zHD'][ww]  # use the vpec corrected redshift for zCMB
  #  print('zCMB', zCMB)
    zHEL = data['zHEL'][ww]
    m_obs = data['m_b_corr'][ww]
    cepheid_distance = data['CEPH_DIST'][ww]
    is_calibrator = data['IS_CALIBRATOR'][ww]
   # print('is_calibrator', is_calibrator)


    filename = default_covmat_file
   # print("Loading covariance from {}".format(filename))

    # The file format for the covariance has the first line as an integer
    # indicating the number of covariance elements, and the the subsequent
    # lines being the elements.
    # This function reads in the file and the nasty for loops trim down the covariance
    # to match the only rows of data that are used for cosmology

    f = open(filename)
    line = f.readline()
    n = int(len(zCMB))
    C = np.zeros((n, n))
    ii = -1
    jj = -1
    mine = 999
    maxe = -999
    for i in range(origlen):
        jj = -1
        if ww[i]:
            ii += 1
        for j in range(origlen):
            if ww[j]:
                jj += 1
            val = float(f.readline())
            if ww[i]:
                if ww[j]:
                    C[ii, jj] = val
    f.close()

    print('Done')
    cov = la.cholesky(C, lower=True, overwrite_a=True)

    return default_data_file, default_covmat_file, zCMB,zHEL, m_obs,  cepheid_distance, is_calibrator, cov