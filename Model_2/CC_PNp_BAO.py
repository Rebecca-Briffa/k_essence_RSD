import emcee
import os
from multiprocessing import Pool, cpu_count
from Likelihoods import *
from scipy.integrate import cumulative_trapezoid

ncpu = cpu_count()


def save_bad_point(theta, reason):
    pid = os.getpid()
    with open(f"bad_points_{pid}.txt", "a") as f:
        f.write(f"{reason} | " + " ".join(map(str, theta)) + "\n")


def log_prior(theta):
    H_0, Om, x0, lambd, M, r_drag = theta
    x0_max = np.sqrt(max(0.0, 2.0 - 2.0 * Om))

    if not (50.0 < H_0 < 100.0):
        return -np.inf
    if not (0.0 < Om < 0.5):
        return -np.inf
    if not (0.0 < x0 < x0_max):
        return -np.inf
    if not (0.0 < lambd < 0.817):
        return -np.inf

    if not (-80.0 < M < 60.0):
        return -np.inf
    if not (100.0 < r_drag < 300.0):
        return -np.inf
    if (1.0 + x0 ** 2 - Om) <= 0.0:
        return -np.inf

    return 0.0


def log_probability(theta):
    H_0, Om, x0, lambd, M, r_drag = theta
    lp = log_prior(theta)
    if not np.isfinite(lp):
        save_bad_point(theta, "prior_failure")
        return -np.inf

    z_int = np.linspace(0.0, 2.4, 800)

    if (1.0 + x0 ** 2 - Om) <= 0.0:
        save_bad_point(theta, "invalid_initial_condition")
        return -np.inf

    try:
        Hz, dH_dz, h_z, x_z, q_z = get_Hz_dHz(H_0, Om, x0, lambd, z_int)
    except Exception:
        save_bad_point(theta, "solver_exception")
        return -np.inf

    if (
            not np.all(np.isfinite(Hz)) or
            not np.all(np.isfinite(dH_dz)) or
            not np.all(np.isfinite(h_z)) or
            not np.all(np.isfinite(x_z)) or
            not np.all(np.isfinite(q_z))
    ):
        save_bad_point(theta, "non_finite_arrays")
        return -np.inf

    if np.any((q_z) <= 0.0):
        save_bad_point(theta, "q_singularity")
        return -np.inf

    Om_z = 1.0 + x_z ** 2 - 3.0 * x_z ** 4 / q_z ** 2
    if np.any(~np.isfinite(Om_z)) or np.any(Om_z <= 0.0):
        save_bad_point(theta, "negative_Om_z")
        return -np.inf

    if np.any((6.0 * x_z ** 2 - q_z ** 2) <= 0.0):
        save_bad_point(theta, "xq_singularity")
        return -np.inf

    if np.any((2 * x_z ** 2 - q_z ** 2) <= 0.0):
        save_bad_point(theta, "cs2_condition")
        return -np.inf

    hz_interp = interpolate.interp1d(
        z_int, Hz, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )
    dhz_interp = interpolate.interp1d(
        z_int, dH_dz, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )

    d_H = c / Hz
    d_C = cumulative_trapezoid(d_H, z_int, initial=0)

   # dD_dz = diff_d_d0(z_int, H_0, Om, f0, hz_interp, dhz_interp)

    # if not np.all(np.isfinite(dD_dz)):
    #     save_bad_point(theta, "non_finite_growth")
    #     return -np.inf

    lp = (
            lp
             + Likelihood_DESI2().loglkl(theta, z_int, Hz, d_C)
           # + Likelihood_fs8().loglkl(theta, z_int, dD_dz, Hz)
            + Likelihood_clocks().loglkl(theta, z_int, Hz)
            + Likelihood_Pantheon_SH0ES().loglkl(theta, z_int, Hz, d_C)
    )

    if np.isnan(lp):
        return -np.inf

    return lp


def build_initial_walkers(nwalkers, eps=1e-4):
    p0 = []
    while len(p0) < nwalkers:
        H0 = np.random.uniform(50.0, 100.0)
        Om = np.random.uniform(0.0, 0.40)
        lambd = np.random.uniform(0.0, 0.817)
        M = np.random.uniform(-80.0, 60.0)
        r_drag = np.random.uniform(100.0, 300.0)

        x0_max = np.sqrt(max(0.0, 2.0 - 2.0 * Om))
        if x0_max <= eps:
            continue

        x0 = np.random.uniform(eps, x0_max)
        theta = [H0, Om, x0, lambd, M, r_drag]

        if log_prior(theta) != 0.0:
            continue
        if not np.isfinite(log_probability(theta)):
            continue

        p0.append(theta)

    return np.array(p0)


ndim, nwalkers = 6, 100
niter = 50000
p0 = build_initial_walkers(nwalkers)

if __name__ == '__main__':

    with Pool(6) as pool:
        moves = [emcee.moves.StretchMove(), emcee.moves.DEMove()]
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, moves=moves, pool=pool
        )

        print('\nRunning burn-in...')
        p0, lp0, _ = sampler.run_mcmc(p0, 500, progress=True)
        n_finite = np.sum(np.isfinite(lp0))
        print(f'Burn-in: {n_finite}/{nwalkers} walkers have finite log prob')
        if n_finite < nwalkers // 2:
            raise RuntimeError('Most walkers still at -inf after burn-in.')
        sampler.reset()

        print('Running production...')

        index = 0
        autocorr = np.empty(niter)
        old_tau = None

        for sample in sampler.sample(p0, iterations=niter, progress=True):
            if sampler.iteration % 100:
                continue
            if sampler.iteration < 500:
                continue

            try:
                tau = sampler.get_autocorr_time(tol=0.01)
            except emcee.autocorr.AutocorrError:
                continue

            if not np.all(np.isfinite(tau)):
                continue

            autocorr[index] = np.mean(tau)
            index += 1

            converged = np.all(tau * 50 < sampler.iteration)
            if old_tau is not None:
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

            print('\n --> old_tau = {0}, new_tau = {1} not convergent!!'.format(old_tau, tau))
            if converged:
                print('\n --> old_tau = {0}, new_tau = {1} convergent!!'.format(old_tau, tau))
                break
            old_tau = tau

        flat_samples = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob(flat=True)

        print("Any NaN in chain:", np.isnan(flat_samples).any())
        print("Any Inf in chain:", np.isinf(flat_samples).any())
        print("Std per parameter:", np.std(flat_samples.reshape(-1, ndim), axis=0))

        np.savetxt('M2_CC_PN_BAO_chain_latest_tight_2710.txt', flat_samples)
        np.savetxt('M2_CC_PN_BAO_log_prob_latest_2710.txt', log_prob)
