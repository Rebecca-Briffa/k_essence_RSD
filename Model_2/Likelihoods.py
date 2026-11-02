from PNplus_data import *
from CC_moresco import *
from pantheon_lehel_data import pantheon_lehel_data
from numpy import loadtxt
from util import *
from scipy.integrate import cumulative_trapezoid
import scipy.linalg as la
import time
import pandas as pd



CC_z, CC_Hz, errHz, zmod, imf, slib, sps, spsooo, cov_mat, inv_cov_mat = CC_moresco()
C_LIGHT = 2.99792458e5


def _model2_params(theta):
    """
    Unpack Model 2 parameter vectors of different lengths.

    Layouts:
      5: [H0, Om, x0, lambd, M]
      6: [H0, Om, x0, lambd, M, r_drag]
      7: [H0, Om, x0, lambd, s80, f0, M]
      8: [H0, Om, x0, lambd, s80, f0, M, r_drag]
    """
    H_0, Om, x0, lambd = theta[:4]
    n = len(theta)
    s80 = f0 = r_drag = None

    if n == 5:
        M = theta[4]
    elif n == 6:
        M, r_drag = theta[4], theta[5]
    elif n == 7:
        s80, f0, M = theta[4], theta[5], theta[6]
    elif n >= 8:
        s80, f0, M, r_drag = theta[4], theta[5], theta[6], theta[7]
    else:
        raise ValueError(f"Unexpected parameter vector length: {n}")

    return H_0, Om, x0, lambd, s80, f0, M, r_drag


class Likelihood_clocks:
    '''
    Cosmic chronometers likelihood
    '''
    def loglkl(self, theta, z_int, Hz):
        H_0, Om, x0, lambd, _, _, M, _ = _model2_params(theta)

        # compute the interpolation at the chosen redshifts
        Hz_inter = interpolate.interp1d(z_int, Hz)
        # obtain theoretical points
        H_z = Hz_inter(CC_z)

        # obtain chi2
        chi2 = 0.

        residual = CC_Hz - H_z
        for i in range(0, len(CC_z)):
            for j in range(0, len(CC_z)):
                chi2 = chi2 + ((residual[i]) * inv_cov_mat[i, j] * (residual[j]))

        return -0.5 * chi2


datafile, covfile, zCMB, zHEL, m_obs, cepheid_distance, is_calibrator, cov = Pantheon_data()

PN_z_sn, PN_m_obs, PN_inv_cov_mat = pantheon_lehel_data()


class Likelihood_Pantheon_Lehel:
    '''
    Pantheon supernova likelihood using Lehel's data files and formulation.
    '''

    def __init__(self):
        self.z_sn = PN_z_sn
        self.m_obs = PN_m_obs
        self.inv_cov_mat = PN_inv_cov_mat
        self.z_grid = np.linspace(0.0, 2.4, 10000)

    def loglkl(self, theta, z_int, h_z):
        H_0, Om, x0, lambd, _, _, M, _ = _model2_params(theta)

        H_val = interpolate.interp1d(
            z_int, h_z, kind='cubic', bounds_error=False, fill_value='extrapolate'
        )
        h_grid = H_val(self.z_grid)

        if np.any(~np.isfinite(h_grid)) or np.any(h_grid <= 0.0):
            return -np.inf

        integral_grid = cumulative_trapezoid(1.0 / h_grid, self.z_grid, initial=0.0)
        integral_sn = np.interp(self.z_sn, self.z_grid, integral_grid)

        dL_model = (C_LIGHT / H_0) * (1.0 + self.z_sn) * integral_sn

        if np.any(~np.isfinite(dL_model)) or np.any(dL_model <= 0.0):
            return -np.inf

        m_model = 5.0 * np.log10(dL_model) + 25.0 + M
        residual = self.m_obs - m_model
        chi2 = residual @ self.inv_cov_mat @ residual

        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2


class Likelihood_Pantheon_SH0ES:
    '''
    Pantheon Supernovae dataset
    '''

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, x0, lambd, _, _, M, _ = _model2_params(theta)

        zcmb = zCMB
        zhel = zHEL
        red_size = zcmb.size
        moduli = zCMB * np.nan

        # obtain the luminosity distance
        cosmo_dist = distances(zcmb, Hz, z_int, d_C)


        # catch error here
        if cosmo_dist == -np.inf:
            chi2 = -np.inf

        else:
            DA = cosmo_dist[0]
            # compute the moduli to be able to compare with data TO CHECK ABOUT SELF.KIND
            f = interpolate.interp1d(zcmb, DA)
            # Here we use the Cepheid host distances as the "theory".
            # This is the 2nd rung of the distance ladder and is what calibrates M
            moduli[np.array(is_calibrator, dtype='bool')] = cepheid_distance[np.array(is_calibrator, dtype='bool')]

            # Actually do the interpolation at the data redshifts
            zcmb = zCMB[~np.array(is_calibrator, dtype='bool')]
            zhel = zHEL[~np.array(is_calibrator, dtype='bool')]
            moduli[~np.array(is_calibrator, dtype='bool')] = 5.0 * np.log10((1.0 + zcmb) * (1.0 + zhel) * np.atleast_1d(f(zcmb))) + 25.

            # Compute the residuals (estimate of distance moduli - exact moduli)
            residuals = np.empty((red_size,))
            mu = m_obs
            # This operation loops over all supernovae!
            # Compute the approximate moduli
            residuals = mu - M

            # Remove from the approximate moduli the one computed from CLASS
            residuals -= moduli



            # Step 2) (Step 1 is done in the init) Solve the triangular system, also time expensive (0.02 seconds)
            residuals = la.solve_triangular(cov, residuals, lower=True, check_finite=False)

            # Finally, compute the chi2 as the sum of the squared residuals
            chi2 = (residuals ** 2).sum()
        return -0.5 * chi2


fs8_file = 'Data/' + 'f_sigma8_updated.csv'
df = pd.read_csv(r'Data/f_sigma8_updated.csv')
df_sort = df.sort_values("z")
# Catalogue row of each redshift-sorted measurement. The covariance uses that order.
order = df_sort.index.to_numpy()

# File stores Cov * 10^3, so entries are O(1). Rescale to (fσ8)^2.
covdata = np.loadtxt('Data/fs8_cov_tot.txt') * 1e-3
covdata = covdata[np.ix_(order, order)]
cov_inv = np.linalg.inv(covdata)



# Likelihood for emcee
class Likelihood_fs8:
    '''
    RSD / f sigma8 likelihood (same implementation as LCDM; numpy arrays)
    '''

    def __init__(self):
        # Use plain numpy arrays (sorted order) to avoid pandas index mismatches
        self.z = np.asarray(df_sort.iloc[:, 0], dtype=float)
        self.fs8 = np.asarray(df_sort.iloc[:, 1], dtype=float)
        self.err = np.asarray(df_sort.iloc[:, 2], dtype=float)

    def loglkl(self, theta, z_int, d, Hz):
        H_0, Om, x0, lambd, s80, f0, M, _ = _model2_params(theta)

        if s80 is None or not np.isfinite(s80):
            return -np.inf

        if -1e-60 in Hz:
            print('WARNING')
            return -np.inf

        od = interpolate.interp1d(z_int, d)
        dD = od(self.z)

        fsigma8_theo = -(1.0 + self.z) * s80 * dD

        if np.any(~np.isfinite(fsigma8_theo)) or np.any(fsigma8_theo < 1e-3):
            return -np.inf

        data_array = self.fs8 - fsigma8_theo
        chi2 = np.dot(np.dot(data_array, cov_inv), data_array)
        return -0.5 * chi2


######################### DESI 2 Likelihood #################################
class Likelihood_DESI2:
    '''
    DESI2 Likelihood https://arxiv.org/pdf/2503.14738
    '''

    def __init__(self):

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # read redshifts and data points
        with open('Data/desi_gaussian_bao_ALL_GCcomb_mean.txt', 'r') as filein:
            for line in filein:
                if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()

                    if this_line[2] == 'DV_over_rs' or 'DM_over_rs' or 'DH_over_rs':
                        self.z = np.append(self.z, float(this_line[0]))
                        self.data = np.append(self.data, float(this_line[1]))
                        self.type = np.append(self.type, int(this_line[3]))

        # read covariance matrix
        self.covdata = np.loadtxt('Data/desi_gaussian_bao_ALL_GCcomb_cov.txt')
        # number of data points
        self.num_points = np.shape(self.z)[0]


    # compute likelihood

    def loglkl(self,theta, z_int, Hz, d_C):
        H_0, Om, x0, lambd, _, _, M, r_drag = _model2_params(theta)
        chi2 = 0.

        if -1e-60 in Hz:
            print('WARNING')
            return -np.inf
        else:
            # compute the interpolation at the chosen redshifts
            Hz_interpolate = interpolate.interp1d(z_int, Hz)
            # obtain the theoretical points
            H_cosmo = Hz_interpolate(self.z)
            # for each point, compute angular distance da,
            # volume distance dv, sound horizon at baryon drag rs_d,
            # theoretical prediction and chi2 contribution
            data_array = np.array([], 'float64')
            theoretical = np.array([], 'float64')
            for i in range(self.num_points):
                #print(i)

                dv = v_weight_eff(self.z[i], Hz, z_int, d_C)
                cosmo_dist = distances(self.z[i], Hz, z_int, d_C)
                dM = cosmo_dist[1]

                # print('vol eff time', t6-t5)
                c_g = 299792458.  # speed of light in m/s
                c = c_g / 1000.  # speed of light in km/s
                dH = c / H_cosmo[i]
                rs = r_drag


                if self.type[i] == 1:
                    theo = dv / rs
                    diff = theo - self.data[i]
                    #print(theo, self.data[i], diff)


                elif self.type[i] == 2:
                    theo = dM / rs
                    diff = theo - self.data[i]
                    #print(theo, self.data[i], diff)


                elif self.type[i] == 3:
                    theo = dH / rs

                    diff = theo - self.data[i]
                   #print(theo, self.data[i], diff)
                # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)


                # appending to the data array in order s.t (DM, H, DM, H,...)
                data_array = np.append(data_array, diff)
                #theoretical = np.append(theoretical, theo)
                #print(len(data_array))



            cov_inv = np.linalg.inv(self.covdata)
            chi2 = np.dot(np.dot(data_array, cov_inv), data_array)

            return -0.5 * chi2





###################### DESI 1 ###############


######################### DESI Likelihood #################################
class Likelihood_DESI:
    '''
    DESI Likelihood https://arxiv.org/pdf/2404.03002
    '''

    def __init__(self):

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # read redshifts and data points
        with open('Data/desi_data.txt', 'r') as filein:
            for line in filein:
                if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()

                    if this_line[2] == 'DV_over_rs' or 'DM_over_rs' or 'DH_over_rs':
                        self.z = np.append(self.z, float(this_line[0]))
                        self.data = np.append(self.data, float(this_line[1]))
                        self.type = np.append(self.type, int(this_line[3]))

        # read covariance matrix
        self.covdata = np.loadtxt('Data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')
        # number of data points
        self.num_points = np.shape(self.z)[0]


    # compute likelihood

    def loglkl(self,theta, z_int, Hz, d_C):
        H_0, Om, B, f0, r_drag= theta
        chi2 = 0.

        if -1e-60 in Hz:
            print('WARNING')
            return -np.inf
        else:
            # compute the interpolation at the chosen redshifts
            Hz_interpolate = interpolate.interp1d(z_int, Hz)
            # obtain the theoretical points
            H_cosmo = Hz_interpolate(self.z)
            # for each point, compute angular distance da,
            # volume distance dv, sound horizon at baryon drag rs_d,
            # theoretical prediction and chi2 contribution
            data_array = np.array([], 'float64')
            theoretical = np.array([], 'float64')
            for i in range(self.num_points):
                #print(i)

                dv = v_weight_eff(self.z[i], Hz, z_int, d_C)
                cosmo_dist = distances(self.z[i], Hz, z_int, d_C)
                dM = cosmo_dist[1]

                # print('vol eff time', t6-t5)
                c_g = 299792458.  # speed of light in m/s
                c = c_g / 1000.  # speed of light in km/s
                dH = c / H_cosmo[i]
                rs = r_drag

                if self.type[i] == 1:
                    theo = dv / rs
                    diff = theo - self.data[i]
                    #print(theo, self.data[i], diff)


                elif self.type[i] == 2:
                    theo = dM / rs
                    diff = theo - self.data[i]
                    #print(theo, self.data[i], diff)


                elif self.type[i] == 3:
                    theo = dH / rs

                    diff = theo - self.data[i]
                   #print(theo, self.data[i], diff)
                # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)


                # appending to the data array in order s.t (DM, H, DM, H,...)
                data_array = np.append(data_array, diff)
                theoretical = np.append(theoretical, theo)
                #print(len(data_array))


            cov_inv = np.linalg.inv(self.covdata)
            chi2 = np.dot(np.dot(data_array, cov_inv), data_array)

            return -0.5 * chi2

