from scipy.integrate.odepack import ODEintWarning

from solver_new import *
import numpy as np
from scipy import integrate
from pandas import read_table
import time
from scipy import interpolate
from scipy.integrate import odeint
import warnings
# Obtain the  comoving sound horizon at the baryon drag epoch
# def r_drag(H_0, Om,hz_interp):
#
#     # defining the redshift at the drag epoch - z_drag- Planck18 Table 2
#     z_drag = 1059.94
#     # defining the upper limit of the integral - the scale factor at the drag epoch - a_drag
#     a_drag = 1 / (1+z_drag)
#
#    # the speed of light in km/s^2
#     c = 2.99792458e8
#     c_km = c / 1000
#     H_z = hz_interp
#
#
#     # defining a function for the integrand
#     def integrand(a):
#         # define z with the integration argument a
#         z = (1/a) - 1
#         # call the Friedmann equation from the solver to obtain H(z)
#         Hz = H_z(z)
#         print(Hz)
#
#         # define h = H0/100
#         h = H_0 /100
#
#         # values obtained from Planck 18
#         Ob_h2 = 0.02237
#         Ob = Ob_h2 / h**2
#
#         Oy_h2 = 2.4697*10**(-5)
#         Oy = Oy_h2 / h**2
#
#         R = (3/4) * (Ob/Oy)*a
#         c_s = 1/(np.sqrt(3*(1 + R)))
#
#         return (c_s / (a**2 * Hz))
#
#     # solving the integration using integration.quad from 0 up to a_drag defined previously
#     integration = integrate.quad(integrand, 0, a_drag)
#     return integration[0]*c_km
# #
# # Obtain an approx for the comoving sound horizon at the baryon drag epoch (https://arxiv.org/pdf/2106.00428.pdf)
# def r_drag(H_0 ,Om):
#
#     h = H_0 / 100
#     O_b = 0.02166/ (h**2)
#     w_b = O_b*h*h
#     w_m = Om*h*h
#
#     a1 = 0.00257366
#     a2 = 0.05032
#     a3 = 0.013
#     a4 = 0.7720642
#     a5 = 0.24346362
#     a6 = 0.00641072
#     a7 = 0.5350899
#     a8 = 32.7525
#     a9 = 0.315473
#
#     rs_den = (a1*(w_b**a2) + (a3*(w_b**a4)*(w_m**a5)) + (a6*w_m**a7))
#     rs = 1/rs_den - (a8/w_m**a9)
#     return rs

# the speed of light in km/s^2
c_g = 299792458.  # speed of light in m/s
c = c_g / 1000.  # speed of light in km/s


# calculates the luminosity distance D_L = D_C * (1+z)
def distances(z, Hz, z_int, d_C):
    if -1e-60 in Hz:
        print('WARNING')
        return -np.inf
    else:

        dC_interpolation = interpolate.interp1d(z_int, d_C)
        # obtaining the function for d_C and then obtaining values at each z
        t12 = time.time()
        Dc = dC_interpolation(z)

        d_L = Dc * (1 + z)

        # calculates angular diameter distance d_A
        d_A = Dc / (1 + z)

        # calculates the comoving angular diameter D_M = D_A * (1+z)
        d_M = d_A * (1+z)
        t13 = time.time()
        # print('distances', t13 - t12, H_0, dH_0, d2H_0, Om, q)
        return  d_A, d_M,  d_L

#
def v_weight_eff(z, Hz, z_int, d_C):
    # compute the interpolation at the chosen redshifts
    Hz_interpolate = interpolate.interp1d(z_int, Hz)
    # obtain the 31 theoretical points
    H_z = Hz_interpolate(z)
    cosmo_dist = distances(z, Hz, z_int, d_C)
    if cosmo_dist == -np.inf:
        return -np.inf
    else:
        d_M = cosmo_dist[1]
        d_V = pow(d_M * d_M * c * z / H_z, 1. / 3.)
        return d_V

# read the matrix for the Supernoave data
def read_matrix(path):
    with open(path, 'r') as text:
        length = int(text.readline())
    matrix = read_table(path).to_numpy().reshape((length, length))
    return matrix


def read_light_curve_parameters(path):

    with open(path, 'r') as text:
        clean_first_line = text.readline()[1:].strip()
        names = [e.strip().replace('3rd', 'third')
                 for e in clean_first_line.split( )]

    lc_parameters = read_table(
        path, sep=' ', names=names, header=0, index_col=False)
    return lc_parameters


# ============================================================
# =============== GROWTH EQUATION (D(z)) =====================
# ============================================================

def D_diff(y, z,  H_0, Om, hz_interp, dhz_interp):
    """
    Differential equation for the linear growth factor D(z).

    Parameters
    ----------
    y : array
        [D, g], where g = dD/dz
    z : float
        Redshift
    params : tuple
        (H0, B, Ω_m)

    Returns
    -------
    dydz : array
        [dD/dz, dg/dz]
    """
    #H0, B_param, Om = params
    D, g = y

    # Interpolate background quantities
    hz = hz_interp(z)
    dhz = dhz_interp(z)

    # Terms in the differential equation
    part1 = -(dhz / hz - 1.0 / (1.0 + z))
    Q = 1.0  # modification factor (could vary with model)
    part2 = 1.5 * Q * (H_0 / hz)**2 * Om * (1.0 + z)

    # Differential system
    dD_dz = g
    dg_dz = part1 * g + part2 * D
    return [dD_dz, dg_dz]


def diff_d_d0(z_int, H_0, Om, f0, hz_interp, dhz_interp):
    """
    Solves for dD/dz as a function of z.

    Returns
    -------
    dD : array
        Derivative of D(z)
    """
    # Same convention as Model 1/2: g(0) = f0 (typically negative, ~ -f_growth)
    y0 = [1.0, f0]  # D(0)=1, g(0)=dD/dz = f0

    sol = odeint(D_diff, y0, z_int, args=(H_0, Om, hz_interp, dhz_interp))
    dD = sol[:, 1]
    return dD

#
# def D_diff(y, z,  H_0, B, Om, hz_interp, dhz_interp):
#     """
#     Differential equation for the linear growth factor D(z).
#
#     Parameters
#     ----------
#     y : array
#         [D, g], where g = dD/dz
#     z : float
#         Redshift
#     params : tuple
#         (H0, B, Ω_m)
#
#     Returns
#     -------
#     dydz : array
#         [dD/dz, dg/dz]
#     """
#     #H0, B_param, Om = params
#     D, g = y
#
#     # Interpolate background quantities
#     hz = hz_interp(z)
#     dhz = dhz_interp(z)
#
#     # Terms in the differential equation
#     part1 = -(dhz / hz - 1.0 / (1.0 + z))
#     Q = 1.0  # modification factor (could vary with model)
#     part2 = 1.5 * Q * (H_0 / hz)**2 * Om * (1.0 + z)
#
#     # Differential system
#     dD_dz = g
#     dg_dz = part1 * g + part2 * D
#     return [dD_dz, dg_dz]
#
# def diff_d_d0(z_int, H_0, B, Om, hz_interp, dhz_interp):
#     """
#     Solves for dD/dz as a function of z using matter-dominated initial conditions at z_init.
#
#     Parameters
#     ----------
#     z_int : array
#         Redshifts where D' is evaluated (e.g., your data redshifts)
#     H_0, B, Om : floats
#         Cosmological parameters
#     hz_interp, dhz_interp : callable
#         Interpolated background H(z) and dH/dz
#     z_init : float
#         Initial redshift for integration (high-z matter dominated)
#     """
#
#
#     # Matter-dominated initial conditions at z_init = 50
#     z_init = 50
#     D_init = 1.0 / (1.0 + z_init)
#     g_init = -1.0 / (1.0 + z_init)**2
#     y0 = [D_init, g_init]
#
#     # Full integration from z_init down to 0
#     z_full = np.linspace(z_init, 0, 1000)
#     sol = odeint(D_diff, y0, z_full, args=(H_0, B, Om, hz_interp, dhz_interp))
#
#     # Extract D and normalize so that D(0)=1
#     D = sol[:, 0] / sol[-1, 0]
#     dDdz = sol[:, 1] / sol[-1, 0]  # derivative scales the same way
#
#     # Interpolate solution at the desired redshifts
#     D_interp = np.interp(z_int, z_full[::-1], dDdz[::-1])  # reverse because odeint integrates from high z down
#     return D_interp
