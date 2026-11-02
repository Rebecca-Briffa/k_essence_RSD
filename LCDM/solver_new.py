import numpy as np
from scipy import optimize
from scipy.integrate import odeint, solve_ivp

# def solver_Ez(z, y, B):
#     """
#     Differential equations for the background evolution.
#
#     Parameters
#     ----------
#     z : float
#         Redshift
#     y : array
#         [h, φ], where h = H/H₀ and φ is an auxiliary field
#     params : tuple
#         (H0, B, Ω_m0)
#
#     Returns
#     -------
#     dydz : array
#         Derivatives [dh/dz, dφ/dz]
#     """
#
#
#     h, phi = y
#
#     # Denominator to avoid singularities
#     denom = (1 + z) * h * (1 - 4 * B * phi**2)
#
#     # Background evolution equations
#     dh_dz = (3.0 * h**2 - 0.5 * phi**2 + (1.0 / 3.0) * B * phi**4) / (2.0 * (1 + z) * h)
#     dphi_dz = 0.0 if abs(denom) < 1e-12 else (-4.0 * B * h * phi**3 + 3.0 * h * phi) / denom
#
#
#     return np.array([dh_dz, dphi_dz])
#
#
# def diff_h_phi(H_0, B, Om, z_eval):
#     """
#     Integrates the background ODE system to obtain h(z) and φ(z).
#
#     Parameters
#     ----------
#     params : tuple
#         (H0, B, Ω_m0)
#     z_eval : array
#         Redshift array for evaluation
#
#     Returns
#     -------
#     (z, h(z), φ(z))
#     """
# #    H0, B_param, Omega_m0 = params
#
#     # Initial conditions (ensure sqrt argument is valid)
#     inner = (1 - 48 * B * (-1 + Om))
#
#
#     phi0 = -0.5 * np.sqrt((1 + np.sqrt(inner)) / B)
#     y0 = [1.0, phi0]  # Initial h(z=0) = 1 (normalized), φ(z=0) = phi0
#
#     # Solve system using BDF method (stiff solver)
#     sol = solve_ivp(
#         lambda zz, yy: solver_Ez(zz, yy, B),
#         [0, 2.4], y0,
#         t_eval=z_eval,
#         method='BDF',
#         rtol=1e-6, atol=1e-12
#     )
#
#     if not sol.success:
#         raise RuntimeError("Background solver failed: " + str(sol.message))
#
#     return sol.t, sol.y[0], sol.y[1]
#
#
# def H_dH_arrays(H_0, B, Om, z_values):
#     """
#     Computes H(z) and dH/dz for a given parameter set.
#
#     Returns
#     -------
#     Hz : array
#         Hubble rate at each z
#     dH_dz : array
#         Derivative of H(z)
#     """
#
#     z_sol, h_z, phi_z = diff_h_phi(H_0, B, Om, z_values)
#
#
#     if len(z_sol) != len(z_values):
#         raise ValueError("Background solution length mismatch.")
#
#     # Convert normalized h(z) to physical H(z)
#     Hz = h_z * H_0
#
#     # Derivative from same model equations wrt z, now having H(z)
#     dH_dz = H_0 * ((3.0 * h_z**2 - 0.5 * phi_z**2 + (1.0 / 3.0) * B * phi_z**4) / (2.0 * (1 + z_values) * h_z))
#
#     return Hz, dH_dz
#


# def H_dH_arrays(H_0, Om, z_values):
#     h = H_0 / 100
#     OR = 4.1534 * (10 ** (-5)) / h ** 2
#     z = z_values
#     sqrt_H = H_0**2*(Om*(1. + z)**3 + OR*(1. + z)**4 + (1. - Om - OR))
#     # print(sqrt_H)
#     # catch error here
#     # catch error here
#     if sqrt_H<0:
#         return -1.e-60
#     else:
#         H = np.sqrt(sqrt_H)
#         dH = H / (1 + z) + (H_0 ** 2 / (2 * (1 + z) * H) * (2 * (OR * (1 + z) ** 4) + Om * (1 + z) ** 3 - 2 * (1 - Om - OR)))
#         return H, dH



def H_dH_arrays(H_0, Om, z_values):
    """
    Computes H(z) and dH/dz for ΛCDM in the same format as the ODE-based model.

    Parameters
    ----------
    H_0 : float
        Hubble constant today
    Om : float
        Matter density parameter Ω_m0
    z_values : array
        Redshift array

    Returns
    -------
    Hz : array
        Hubble rate at each z
    dH_dz : array
        Derivative dH/dz at each z
    """

    z = np.array(z_values)

    # Radiation density parameter
    h = H_0 / 100.0
    OR = 4.1534e-5 / h**2

    # Compute H(z)^2
    Hz_sq = H_0**2 * (
        Om * (1 + z)**3 +
        OR * (1 + z)**4 +
        (1 - Om - OR)
    )

    # Ensure numerical safety (vectorized)
    if np.any(Hz_sq <= 0):
        raise ValueError("Encountered non-physical H^2 <= 0 in LCDM.")

    # Hubble parameter
    Hz = np.sqrt(Hz_sq)

    # dH/dz (consistent analytic derivative)
    dH_dz = (H_0**2 / (2 * Hz)) * (
        3 * Om * (1 + z)**2 +
        4 * OR * (1 + z)**3
    )
    #dH = H / (1 + z) + ( H_0 ** 2 / (2 * (1 + z) * H) * (2 * (OR * (1 + z) ** 4) + Om * (1 + z) ** 3 - 2 * (1 - Om - OR)))

    return Hz, dH_dz