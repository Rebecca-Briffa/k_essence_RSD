# import numpy as np
# from scipy import optimize
# from scipy.integrate import odeint, solve_ivp
#
# def solver_system(z, y, omegam, lambd):
#     """
#     Defines the system of differential equations for the background evolution.
#
#     Parameters
#     ----------
#     z : float
#         Redshift
#     y : array
#         Dynamical variables [h, x, q]
#         - h = H/H0 (dimensionless Hubble parameter)
#         - x, q = auxiliary dynamical variables
#     omegam : float
#         Present-day matter density parameter Ω_m0
#     lambd : float
#         Model parameter (e.g. potential slope)
#
#     Returns
#     -------
#     dydz : array
#         Derivatives [dh/dz, dx/dz, dq/dz]
#     """
#
#     h, x, q = y
#
#     # --- Denominators (used to avoid numerical instabilities) ---
#     denom_x = (1 + z) * q**2 * (6 * x**2 - q**2)
#     denom_q = (1 + z) * q
#
#     # --- Hubble evolution equation ---
#     dh_dz = -(3 * h * (-x**4 - q**2 + x**2 * q**2)) / (2 * (1 + z) * q**2)
#
#     # --- Evolution of auxiliary variable x ---
#     dx_dz = 0.0 if abs(denom_x) < 1e-12 else -3 * (
#         6*x**7 + 2*x**3*q**2
#         - np.sqrt(6)*lambd*x**4*q**2
#         - 7*x**5*q**2 + x*q**4 + x**3*q**4
#     ) / (2 * denom_x)
#
#     # --- Evolution of auxiliary variable q ---
#     dq_dz = 0.0 if abs(denom_q) < 1e-12 else -(
#         3*x**4 + 3*q**2
#         - np.sqrt(6)*lambd*x*q**2
#         - 3*x**2*q**2
#     ) / (2 * denom_q)
#
#     return np.array([dh_dz, dx_dz, dq_dz])
# def solve_background(H0, omegam, x0, lambd, z_eval):
#     """
#     Solves the background system for given cosmological parameters.
#
#     Parameters
#     ----------
#     H0 : float
#         Hubble constant today
#     omegam : float
#         Present-day matter density parameter Ω_m0
#     x0 : float
#         Initial value of auxiliary field x at z = 0
#     lambd : float
#         Model parameter
#     z_eval : array
#         Redshift values where the solution is evaluated
#
#     Returns
#     -------
#     z : array
#         Redshift grid
#     h(z) : array
#         Dimensionless Hubble parameter
#     x(z), q(z) : arrays
#         Auxiliary field solutions
#     """
#
#     # --- Physical consistency check for initial condition ---
#     if (1 + x0**2 - omegam) <= 0:
#         raise ValueError("Invalid initial condition: sqrt argument negative")
#
#     # --- Initial condition for q derived from constraint equation ---
#     q0 = np.sqrt(3) * x0**2 / np.sqrt(1 + x0**2 - omegam)
#
#     # Initial state vector at z = 0
#     y0 = [1.0, x0, q0]
#
#     # --- Solve the system (stiff solver) ---
#     sol = solve_ivp(
#         lambda zz, yy: solver_system(zz, yy, omegam, lambd),
#         [0, max(z_eval)],
#         y0,
#         t_eval=z_eval,
#         method='BDF',
#         rtol=1e-6,
#         atol=1e-12
#     )
#
#     # --- Check solver success ---
#     if not sol.success:
#         raise RuntimeError("Background solver failed")
#
#     return sol.t, sol.y[0], sol.y[1], sol.y[2]
#
#
# def get_Hz_dHz(H0, omegam, x0, lambd, z_values):
#     """
#     Computes H(z) and dH/dz from the background solution.
#
#     Returns
#     -------
#     Hz : array
#         Hubble rate
#     dHz_dz : array
#         Derivative of H(z)
#     """
#
#     # Solve system
#     z_sol, h_z, x_z, q_z = solve_background(H0, omegam, x0, lambd, z_values)
#
#     # --- Compute dh/dz using SAME equation as solver ---
#     denom = 2 * (1 + z_values) * q_z ** 2
#     safe = np.abs(denom) > 1e-12
#
#     dh_dz = np.zeros_like(h_z)
#     dh_dz[safe] = -(3 * h_z[safe] * (-x_z[safe] ** 4 - q_z[safe] ** 2 + x_z[safe] ** 2 * q_z[safe] ** 2)) / denom[safe]
#
#     # Convert to physical units
#     Hz = H0 * h_z
#     dHz_dz = H0 * dh_dz
#
#     return Hz, dHz_dz, h_z, x_z, q_z


import numpy as np
from scipy.integrate import solve_ivp


# =========================================================
# 🔹 ODE SYSTEM
# =========================================================
def solver_system(z, y, omegam, lambd):

    h, x, q = y

    # --- Denominators ---
    denom_x = (1 + z) * q**2 * (6 * x**2 - q**2)
    denom_q = (1 + z) * q
    denom_h = 2 * (1 + z) * q**2

    # --- Hard fail inside solver (prevents instability propagation)
    if (
        abs(denom_x) < 1e-12 or
        abs(denom_q) < 1e-12 or
        abs(denom_h) < 1e-12 or
        not np.isfinite(h) or
        not np.isfinite(x) or
        not np.isfinite(q)
    ):
        return [np.nan, np.nan, np.nan]

    # --- Equations ---
    dh_dz = -(3 * h * (-x**4 - q**2 + x**2 * q**2)) / denom_h

    dx_dz = -3 * (
        6*x**7 + 2*x**3*q**2
        - np.sqrt(6)*lambd*x**4*q**2
        - 7*x**5*q**2 + x*q**4 + x**3*q**4
    ) / (2 * denom_x)

    dq_dz = -(
        3*x**4 + 3*q**2
        - np.sqrt(6)*lambd*x*q**2
        - 3*x**2*q**2
    ) / (2 * denom_q)

    return np.array([dh_dz, dx_dz, dq_dz])


# =========================================================
# 🔹 BACKGROUND SOLVER
# =========================================================
def solve_background(H0, omegam, x0, lambd, z_eval):

    # --- Physical consistency ---
    if (1 + x0**2 - omegam) <= 0:
        return np.inf, None, None, None

    # --- Initial q ---
    q0 = np.sqrt(3) * x0**2 / np.sqrt(1 + x0**2 - omegam)

    if not np.isfinite(q0):
        return np.inf, None, None, None

    y0 = [1.0, x0, q0]

    # --- Solve system ---
    sol = solve_ivp(
        lambda zz, yy: solver_system(zz, yy, omegam, lambd),
        [0, max(z_eval)],
        y0,
        t_eval=z_eval,
        method='BDF',
        rtol=1e-6,
        atol=1e-12
    )

    if not sol.success:
        return np.inf, None, None, None

    h_z, x_z, q_z = sol.y

    # --- Numerical sanity ---
    if (
        np.any(~np.isfinite(h_z)) or
        np.any(~np.isfinite(x_z)) or
        np.any(~np.isfinite(q_z))
    ):
        return np.inf, None, None, None

    return sol.t, h_z, x_z, q_z


# =========================================================
# 🔹 H(z) AND dH/dz
# =========================================================
def get_Hz_dHz(H0, omegam, x0, lambd, z_values):

    z_sol, h_z, x_z, q_z = solve_background(H0, omegam, x0, lambd, z_values)

    # --- Propagated failure ---
    if isinstance(z_sol, float) and np.isinf(z_sol):
        return np.inf, np.inf, None, None, None

    # --- Denominator checks (STRICT) ---
    denom_h = 2 * (1 + z_values) * q_z**2

    if np.any(np.abs(denom_h) < 1e-12):
        return np.inf, np.inf, None, None, None

    # --- Compute derivative ---
    dh_dz = -(3 * h_z * (-x_z**4 - q_z**2 + x_z**2 * q_z**2)) / denom_h

    Hz = H0 * h_z
    dHz_dz = H0 * dh_dz

    # --- Final safety ---
    if (
        np.any(~np.isfinite(Hz)) or
        np.any(~np.isfinite(dHz_dz))
    ):
        return np.inf, np.inf, None, None, None

    return Hz, dHz_dz, h_z, x_z, q_z


def H_dH_arrays(H0, omegam, x0, lambd, z_values):
    """
    Convenience wrapper returning only H(z) and dH/dz arrays.
    """
    Hz, dHz_dz, _, _, _ = get_Hz_dHz(H0, omegam, x0, lambd, z_values)
    return Hz, dHz_dz
